/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/python/pjrt_ifrt/pjrt_layout.h"

#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char PjRtLayout::ID = 0;

namespace {

// Performs a basic check on whether `layout` may be converted to `PjRtLayout`.
absl::Status CheckSupportedByteStrides(DType dtype, const Shape& shard_shape,
                                       const ByteStridesLayout& layout) {
  std::optional<int> byte_size = dtype.byte_size();
  if (!byte_size.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("ByteStridesLayout expects DType that has a byte size "
                     "if shard_shape dimensions are not empty, but got dtype=",
                     dtype));
  }
  absl::Span<const int64_t> dimensions = shard_shape.dims();
  absl::Span<const int64_t> byte_strides = layout.byte_strides();
  if (dimensions.size() != byte_strides.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("ByteStridesLayout has ", byte_strides.size(),
                     " dimensions, but shard_shape has ", dimensions.size(),
                     " dimensions."));
  }
  for (int64_t i = 0; i < dimensions.size(); ++i) {
    if (byte_strides[i] < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Only non-negative byte strides are supported: layout=", layout));
    }
  }
  return absl::OkStatus();
}

// Checks if a computed `minor_to_major` is valid. `dtype` must have a byte
// size.
absl::Status CheckValidMinorToMajor(DType dtype, const Shape& shard_shape,
                                    const ByteStridesLayout& layout,
                                    absl::Span<const int64_t> minor_to_major) {
  std::optional<int> byte_size = dtype.byte_size();
  absl::Span<const int64_t> dimensions = shard_shape.dims();
  absl::Span<const int64_t> byte_strides = layout.byte_strides();
  int64_t byte_stride = *byte_size;
  for (int64_t d : minor_to_major) {
    if (dimensions[d] != 1 && byte_strides[d] != byte_stride) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Only dense byte strides are supported; i.e., byte "
          "striding represents a transposition of the underlying dense "
          "buffer but not broadcasting: shard_shape=",
          shard_shape, ", layout=", layout));
    }
    byte_stride *= dimensions[d];
  }
  return absl::OkStatus();
}

// Converts a `ByteStridesLayout` into a `PjRtLayout`. Only supports a dense
// layout with non-negative byte strides.
absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> ToPjRtLayout(
    DType dtype, const Shape& shard_shape, const ByteStridesLayout& layout) {
  // Within PjRt-IFRT, it is a convention to use an empty `PjRtLayout` for
  // `DType::kToken`. In XLA, `TOKEN` should not populate `xla::Layout`.
  if (dtype.kind() == DType::kToken) {
    if (!shard_shape.dims().empty()) {
      return absl::InvalidArgumentError(
          "Token is expected to have an empty shard shape.");
    }
    if (!layout.byte_strides().empty()) {
      return absl::InvalidArgumentError(
          "Token is expected to have an empty byte strides.");
    }
    return std::make_shared<xla::PjRtLayout>(xla::Layout());
  }

  TF_RETURN_IF_ERROR(CheckSupportedByteStrides(dtype, shard_shape, layout));

  absl::Span<const int64_t> dimensions = shard_shape.dims();
  absl::Span<const int64_t> byte_strides = layout.byte_strides();

  std::vector<int64_t> minor_to_major(dimensions.size());
  // Begin with a major-to-minor layout that is likely the most common.
  std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);

  // Find minor-to-major only if there is no zero dimension size because
  // minor-to-major is irrelevant with any zero dimension size.
  if (absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::c_sort(minor_to_major, [&](int a, int b) {
      if (byte_strides[a] < byte_strides[b]) {
        return true;
      }
      if (byte_strides[a] > byte_strides[b]) {
        return false;
      }
      return dimensions[a] == 1 && dimensions[b] != 1;
    });
    TF_RETURN_IF_ERROR(
        CheckValidMinorToMajor(dtype, shard_shape, layout, minor_to_major));
  }
  return std::make_shared<xla::PjRtLayout>(xla::Layout(minor_to_major));
}

}  // namespace

absl_nonnull std::unique_ptr<PjRtLayout> PjRtLayout::Create(
    std::shared_ptr<const xla::PjRtLayout> pjrt_layout) {
  return absl::WrapUnique<PjRtLayout>(new PjRtLayout(std::move(pjrt_layout)));
}

absl::StatusOr<std::pair<int64_t, int64_t>> PjRtLayout::ByteBound(
    DType dtype, const Shape& shard_shape) const {
  TF_ASSIGN_OR_RETURN(auto xla_primitive_type, ToPrimitiveType(dtype));
  auto xla_shape =
      xla::ShapeUtil::MakeShape(xla_primitive_type, shard_shape.dims());
  *xla_shape.mutable_layout() = pjrt_layout_->xla_layout();
  return std::pair<int64_t, int64_t>(0, xla::ShapeUtil::ArraySize(xla_shape));
}

bool PjRtLayout::operator==(const Layout& other) const {
  if (this == &other) {
    return true;
  }
  if (const auto* other_pjrt_layout = llvm::dyn_cast<PjRtLayout>(&other)) {
    return *pjrt_layout_ == static_cast<const xla::PjRtLayout&>(
                                *other_pjrt_layout->pjrt_layout_);
  }
  return false;
}

uint64_t PjRtLayout::hash() const { return absl::HashOf(*pjrt_layout_); };

std::string PjRtLayout::ToString() const {
  return absl::StrCat("PjRtLayout(pjrt_layout=", pjrt_layout_->ToString(), ")");
}

absl::StatusOr<absl_nonnull std::shared_ptr<const xla::PjRtLayout>>
ToPjRtLayout(DType dtype, const Shape& shard_shape,
             const ConcreteLayout& layout) {
  if (const auto* pjrt_layout = llvm::dyn_cast<PjRtLayout>(&layout)) {
    return pjrt_layout->pjrt_layout();
  }
  if (const auto* byte_strides_layout =
          llvm::dyn_cast<ByteStridesLayout>(&layout)) {
    return ToPjRtLayout(dtype, shard_shape, *byte_strides_layout);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported layout type: ", layout));
}

absl::StatusOr<absl_nonnull std::shared_ptr<const xla::PjRtLayout>>
ToPjRtLayout(DType dtype, const Shape& shard_shape,
             const absl::StatusOr<LayoutRef>& layout) {
  TF_RETURN_IF_ERROR(layout.status());
  TF_ASSIGN_OR_RETURN(std::shared_ptr<ConcreteLayout> concrete_layout,
                      (*layout)->GetConcrete());
  return ToPjRtLayout(dtype, shard_shape, *concrete_layout);
}

}  // namespace ifrt
}  // namespace xla
