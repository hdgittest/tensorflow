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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/layout_serdes.pb.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

// Serialization/deserialization for `DefaultLayout`.
class DefaultLayoutSerDes
    : public llvm::RTTIExtends<DefaultLayoutSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::DefaultLayout";
  }

  absl::StatusOr<std::string> Serialize(
      Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const auto* device_default_layout =
        llvm::cast<DefaultLayout>(&serializable);
    DefaultLayoutProto proto;
    *proto.mutable_dtype() = device_default_layout->dtype().ToProto();
    *proto.mutable_shard_shape() =
        device_default_layout->shard_shape().ToProto();
    proto.set_device_id(device_default_layout->device()->Id().value());
    if (device_default_layout->memory_kind().memory_kind().has_value()) {
      proto.set_memory_kind(
          std::string(*device_default_layout->memory_kind().memory_kind()));
    }
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    const auto* deserialize_layout_options =
        llvm::cast<DeserializeLayoutOptions>(options.get());

    DefaultLayoutProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized DefaultLayout");
    }
    TF_ASSIGN_OR_RETURN(DType dtype, DType::FromProto(proto.dtype()));
    TF_ASSIGN_OR_RETURN(Shape shard_shape,
                        Shape::FromProto(proto.shard_shape()));
    TF_ASSIGN_OR_RETURN(Device * device,
                        deserialize_layout_options->client->LookupDevice(
                            DeviceId(proto.device_id())));
    MemoryKind memory_kind;
    if (proto.has_memory_kind()) {
      memory_kind = MemoryKind(proto.memory_kind());
    }
    return DefaultLayout::Create(dtype, std::move(shard_shape), device,
                                 memory_kind);
  }

  static char ID;  // NOLINT
};

// Serialization/deserialization for `ByteStridesLayout`.
class ByteStridesLayoutSerDes
    : public llvm::RTTIExtends<ByteStridesLayoutSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::ByteStridesLayout";
  }

  absl::StatusOr<std::string> Serialize(
      Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const auto* byte_strides_layout =
        llvm::cast<ByteStridesLayout>(&serializable);
    const auto& byte_strides = byte_strides_layout->byte_strides();
    ByteStridesLayoutProto proto;
    proto.mutable_byte_strides()->Reserve(byte_strides.size());
    for (int64_t byte_stride : byte_strides) {
      proto.add_byte_strides(byte_stride);
    }
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    ByteStridesLayoutProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized ByteStridesLayout");
    }
    return ByteStridesLayout::Create(ByteStridesLayout::ByteStrides(
        proto.byte_strides().begin(), proto.byte_strides().end()));
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char DefaultLayoutSerDes::ID = 0;      // NOLINT
[[maybe_unused]] char ByteStridesLayoutSerDes::ID = 0;  // NOLINT

// clang-format off
bool register_device_default_layout_serdes = ([]{
  RegisterSerDes<DefaultLayout>(
      std::make_unique<DefaultLayoutSerDes>());
}(), true);
bool register_byte_strides_layout_serdes = ([]{
  RegisterSerDes<ByteStridesLayout>(
      std::make_unique<ByteStridesLayoutSerDes>());
}(), true);
// clang-format on

}  // namespace
}  // namespace ifrt
}  // namespace xla
