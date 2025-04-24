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

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/pjrt_ifrt/basic_string_array_layout.h"
#include "xla/python/pjrt_ifrt/basic_string_array_layout_serdes.pb.h"

namespace xla {
namespace ifrt {
namespace {

// Serialization/deserialization for `BasicStringArrayLayout`.
class BasicStringArrayLayoutSerDes
    : public llvm::RTTIExtends<BasicStringArrayLayoutSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::BasicStringArrayLayout";
  }

  absl::StatusOr<std::string> Serialize(
      Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const auto* basic_string_layout =
        llvm::cast<BasicStringArrayLayout>(&serializable);
    BasicStringArrayLayoutProto proto;
    // Do not store the layout source because it is always device default.
    proto.set_num_elements(basic_string_layout->num_elements());
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    BasicStringArrayLayoutProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized BasicStringArrayLayout");
    }
    return BasicStringArrayLayout::Create(proto.num_elements());
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char BasicStringArrayLayoutSerDes::ID = 0;  // NOLINT

// clang-format off
bool register_basic_string_array_layout_serdes = ([]{
  RegisterSerDes<BasicStringArrayLayout>(
      std::make_unique<BasicStringArrayLayoutSerDes>());
}(), true);
// clang-format on

}  // namespace
}  // namespace ifrt
}  // namespace xla
