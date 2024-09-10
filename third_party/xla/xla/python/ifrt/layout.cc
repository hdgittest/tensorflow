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

#include "xla/python/ifrt/layout.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char Layout::ID = 0;
char ConcreteLayout::ID = 0;
char DefaultLayout::ID = 0;
char ByteStridesLayout::ID = 0;
char DeserializeLayoutOptions::ID = 0;

bool Layout::is_default() const { return llvm::isa<DefaultLayout>(this); }

absl::StatusOr<absl_nonnull std::unique_ptr<Layout>> Layout::FromProto(
    Client* client, const LayoutProto& layout_proto) {
  return Deserialize<Layout>(
      layout_proto.serialized_layout(),
      std::make_unique<DeserializeLayoutOptions>(client));
}

absl::StatusOr<LayoutProto> Layout::ToProto() const {
  LayoutProto layout_proto;
  // `const_cast<>` is used because `Serialize()` requires a non-const object
  // even though it makes read-only access.
  TF_ASSIGN_OR_RETURN(*layout_proto.mutable_serialized_layout(),
                      Serialize(const_cast<Layout&>(*this),
                                /*options=*/nullptr));
  return layout_proto;
}

absl_nonnull std::unique_ptr<DefaultLayout> DefaultLayout::Create(
    DType dtype, Shape shard_shape, Device* device, MemoryKind memory_kind) {
  return absl::WrapUnique<DefaultLayout>(
      new DefaultLayout(dtype, std::move(shard_shape), device, memory_kind));
}

absl::StatusOr<absl_nonnull std::unique_ptr<DefaultLayout>>
DefaultLayout::Create(DType dtype, const Shape& shape,
                      const Sharding& sharding) {
  if (sharding.devices()->size() == 0) {
    return absl::InvalidArgumentError("Sharding has no devices.");
  }
  TF_ASSIGN_OR_RETURN(Shape shard_shape, sharding.GetShardShape(shape));
  return Create(dtype, std::move(shard_shape),
                sharding.devices()->devices().front(), sharding.memory_kind());
}

absl::StatusOr<ConcreteLayoutRef> DefaultLayout::GetConcrete() const {
  return absl::UnimplementedError("Not implemented.");
  // TODO(hyeontaek): Implement.
  //
  // The following is a tentative implementation. It requires making
  // `Client::GetDefaultLayout()` return `std::shared_ptr<ConcreteLayout>` to
  // avoid cicular dependency between `xla::ifrt::DefaultLayout` and
  // `xla::ifrt::PjRtLayout`.
  //
  // absl::MutexLock lock(&mutex_);
  // if (cached_concrete_layout_ == nullptr) {
  //   TF_ASSIGN_OR_RETURN(
  //       cached_concrete_layout_,
  //       device_->client()->GetDefaultLayout(dtype_, shard_shape_.dims(),
  //                                           device_, memory_kind_));
  // }
  // return cached_concrete_layout_;
}

CustomLayoutRef DefaultLayout::GetIfCustom() const { return nullptr; }

bool DefaultLayout::operator==(const Layout& other) const {
  if (this == &other) {
    return true;
  }
  // TODO(hyeontaek): Relax `device_` equality check so that it is true if two
  // devices are in the same collection of devices that use the sam default
  // layout calculation rule.
  if (const auto* other_default_layout =
          llvm::dyn_cast<DefaultLayout>(&other)) {
    return dtype_ == other_default_layout->dtype_ &&
           shard_shape_ == other_default_layout->shard_shape_ &&
           device_ == other_default_layout->device_ &&
           memory_kind_ == other_default_layout->memory_kind_;
  }
  return false;
}

uint64_t DefaultLayout::hash() const {
  // TODO(hyeontaek): Relax `device_` equality check and make hashing not
  // directly sensitive to `device_`. See operator==().
  return absl::HashOf(dtype_, shard_shape_, device_->Id(), memory_kind_);
};

std::string DefaultLayout::ToString() const {
  return absl::StrCat("DefaultLayout(dtype=", dtype_,
                      ",shard_shape=", shard_shape_, ",device=", *device_,
                      ",memory_kind=", memory_kind_, ")");
}

absl::StatusOr<int64_t> ConcreteLayout::ByteSize(DType dtype,
                                                 const Shape& shape) const {
  TF_ASSIGN_OR_RETURN(const auto bound, ByteBound(dtype, shape));
  return bound.second - bound.first;
}

absl::StatusOr<ConcreteLayoutRef> ConcreteLayout::GetConcrete() const {
  // `const_cast<>` is used to return `ConcreteLayoutRef` from a const method
  // calling `shared_from_this()`, which would only return
  // `std::shared_ptr<const ConcreteLayoutRef>`.
  return const_cast<ConcreteLayout*>(this)->shared_from_this();
}

CustomLayoutRef ConcreteLayout::GetIfCustom() const {
  // `const_cast<>` is used to return `ConcreteLayoutRef` from a const method
  // calling `shared_from_this()`, which would only return
  // `std::shared_ptr<const ConcreteLayoutRef>`.
  return const_cast<ConcreteLayout*>(this)->shared_from_this();
}

absl_nonnull std::unique_ptr<ByteStridesLayout> ByteStridesLayout::Create(
    ByteStrides byte_strides) {
  return absl::WrapUnique<ByteStridesLayout>(
      new ByteStridesLayout(std::move(byte_strides)));
}

absl::StatusOr<absl_nonnull std::unique_ptr<ByteStridesLayout>>
ByteStridesLayout::CreateCompactMajorToMinor(DType dtype,
                                             const Shape& shard_shape) {
  ByteStrides byte_strides(shard_shape.dims().size());
  if (!shard_shape.dims().empty()) {
    std::optional<int> byte_size = dtype.byte_size();
    if (!byte_size.has_value()) {
      return absl::InvalidArgumentError(
          absl::StrCat("ByteStridesLayout expects DType that has a byte size "
                       "if shape dimensions are not empty, but got dtype=",
                       dtype));
    }
    byte_strides[byte_strides.size() - 1] = *byte_size;
    for (int i = byte_strides.size() - 1; i > 0; --i) {
      byte_strides[i - 1] = byte_strides[i] * shard_shape.dims()[i];
    }
  }
  return Create(std::move(byte_strides));
}

absl::StatusOr<std::pair<int64_t, int64_t>> ByteStridesLayout::ByteBound(
    DType dtype, const Shape& shard_shape) const {
  if (byte_strides_.size() != shard_shape.dims().size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("ByteStridesLayout expects Shape with the same number of "
                     "dimensions, but got shard_shape=",
                     shard_shape));
  }
  auto bit_size = dtype.bit_size();
  if (!bit_size.has_value()) {
    return std::pair<int64_t, int64_t>(0, 0);
  }

  int64_t padded_byte_size = (*bit_size + 7) / 8;
  int64_t first_element = 0;
  int64_t last_element = 0;
  for (int i = 0; i < byte_strides_.size(); ++i) {
    if (shard_shape.dims()[i] == 0) {
      return std::pair<int64_t, int64_t>(0, 0);
    }
    if (byte_strides_[i] >= 0) {
      last_element += byte_strides_[i] * (shard_shape.dims()[i] - 1);
    } else {
      first_element += byte_strides_[i] * (shard_shape.dims()[i] - 1);
    }
  }
  return std::make_pair(first_element, last_element + padded_byte_size);
}

bool ByteStridesLayout::operator==(const Layout& other) const {
  if (this == &other) {
    return true;
  }
  if (const auto* other_byte_strides_layout =
          llvm::dyn_cast<ByteStridesLayout>(&other)) {
    return byte_strides_ == other_byte_strides_layout->byte_strides_;
  }
  return false;
}

uint64_t ByteStridesLayout::hash() const {
  return absl::HashOf(byte_strides_);
};

std::string ByteStridesLayout::ToString() const {
  return absl::StrCat("ByteStridesLayout(byte_strides=[",
                      absl::StrJoin(byte_strides_, ","), "])");
}

}  // namespace ifrt
}  // namespace xla
