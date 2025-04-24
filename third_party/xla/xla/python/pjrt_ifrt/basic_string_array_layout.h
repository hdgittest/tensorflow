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

#ifndef XLA_PYTHON_PJRT_IFRT_BASIC_STRING_ARRAY_LAYOUT_H_
#define XLA_PYTHON_PJRT_IFRT_BASIC_STRING_ARRAY_LAYOUT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {

// Describes the layout of a `BasicStringArray`. It always uses a dense
// major-to-minor layout. It is agnostic of shapes except that it tracks the
// total number of elements in an array.
class BasicStringArrayLayout final
    : public llvm::RTTIExtends<BasicStringArrayLayout, ConcreteLayout> {
 public:
  // Creates a BasicStringArrayLayout. Currently, all BasicStringArrayLayouts
  // are considered to be device default layouts.
  static absl_nonnull std::unique_ptr<BasicStringArrayLayout> Create(
      int64_t num_elements);

  ~BasicStringArrayLayout() override = default;

  int64_t num_elements() const { return num_elements_; }

  // Layout implementation.

  absl::StatusOr<std::pair<int64_t, int64_t>> ByteBound(
      DType dtype, const Shape& shard_shape) const override;

  bool operator==(const Layout& other) const override;

  static char ID;  // NOLINT

 private:
  explicit BasicStringArrayLayout(int64_t num_elements)
      : num_elements_(num_elements) {}

  uint64_t hash() const override;
  std::string ToString() const override;

  int64_t num_elements_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_BASIC_STRING_ARRAY_LAYOUT_H_
