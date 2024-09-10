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

#include "xla/python/pjrt_ifrt/basic_string_array_layout.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {

char BasicStringArrayLayout::ID = 0;

absl_nonnull std::unique_ptr<BasicStringArrayLayout>
BasicStringArrayLayout::Create(int64_t num_elements) {
  return absl::WrapUnique<BasicStringArrayLayout>(
      new BasicStringArrayLayout(num_elements));
}

absl::StatusOr<std::pair<int64_t, int64_t>> BasicStringArrayLayout::ByteBound(
    DType dtype, const Shape& shard_shape) const {
  if (dtype != DType(DType::kString)) {
    return absl::InvalidArgumentError(
        absl::StrCat("BasicStringArrayLayout expects DType::kString, but got "
                     "dtype=",
                     dtype.DebugString()));
  }
  if (shard_shape.num_elements() != num_elements_) {
    return absl::InvalidArgumentError(
        absl::StrCat("BasicStringArrayLayout expects Shape with the same "
                     "number of elements, but got shard_shape=",
                     shard_shape.DebugString()));
  }
  return std::pair<int64_t, int64_t>(0,
                                     num_elements_ * sizeof(absl::string_view));
}

bool BasicStringArrayLayout::operator==(const Layout& other) const {
  if (const auto* other_layout =
          llvm::dyn_cast<BasicStringArrayLayout>(&other)) {
    return num_elements_ == other_layout->num_elements_;
  }
  return false;
}

uint64_t BasicStringArrayLayout::hash() const {
  return absl::HashOf(num_elements_);
};

std::string BasicStringArrayLayout::ToString() const {
  return absl::StrCat("BasicStringArrayLayout(num_elements=", num_elements_,
                      ")");
}

}  // namespace ifrt
}  // namespace xla
