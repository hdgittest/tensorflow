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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_LAYOUT_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_LAYOUT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {

// Wraps around `xla::PjRtLayout` as an IFRT Layout.
//
// Compatibility note: While `xla::PjRtLayout` may accept take an arbitrary
// `xla::Layout`, we strongly suggest using only a small subset of `xla::Layout`
// features (`minor_to_major`, `tiles`, and `element_size_in_bits`) that are
// approved for use in PjRt C API and less commonly used features.
class PjRtLayout final
    : public llvm::RTTIExtends<xla::ifrt::PjRtLayout, ConcreteLayout> {
 public:
  // Creates a PjRtLayout.
  static absl_nonnull std::unique_ptr<PjRtLayout> Create(
      absl_nonnull std::shared_ptr<const xla::PjRtLayout> pjrt_layout);

  ~PjRtLayout() override = default;

  absl_nonnull const std::shared_ptr<const xla::PjRtLayout>& pjrt_layout()
      const {
    return pjrt_layout_;
  }

  // Layout implementation.

  absl::StatusOr<std::pair<int64_t, int64_t>> ByteBound(
      DType dtype, const Shape& shard_shape) const override;

  bool operator==(const Layout& other) const override;

  static char ID;  // NOLINT

 private:
  explicit PjRtLayout(std::shared_ptr<const xla::PjRtLayout> pjrt_layout)
      : pjrt_layout_(std::move(pjrt_layout)) {}

  uint64_t hash() const override;
  std::string ToString() const override;

  absl_nonnull std::shared_ptr<const xla::PjRtLayout> pjrt_layout_;
};

// Converts IFRT `ConcreteLayout` into `xla::PjRtLayout`. Only supports IFRT
// `PjRtLayout` and `ByteStridesLayout` as input. `ByteStridesLayout` only
// supports a dense layout with non-negative byte strides.
absl::StatusOr<absl_nonnull std::shared_ptr<const xla::PjRtLayout>>
ToPjRtLayout(DType dtype, const Shape& shard_shape,
             const ConcreteLayout& layout);

// Converts IFRT `Layout` into `xla::PjRtLayout`. Only supports IFRT
// `PjRtLayout` and `ByteStridesLayout` as input. `ByteStridesLayout` only
// supports a dense layout with non-negative byte strides.
absl::StatusOr<absl_nonnull std::shared_ptr<const xla::PjRtLayout>>
ToPjRtLayout(DType dtype, const Shape& shard_shape,
             const absl::StatusOr<LayoutRef>& layout);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_LAYOUT_H_
