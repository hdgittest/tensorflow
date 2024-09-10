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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::HasSubstr;
using ::testing::Pair;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(BasicStringArrayLayoutTest, Create) {
  EXPECT_FALSE(BasicStringArrayLayout::Create(3)->is_default());
  EXPECT_EQ(BasicStringArrayLayout::Create(3)->num_elements(), 3);
}

TEST(BasicStringArrayLayoutTest, GetConcrete) {
  std::shared_ptr<Layout> layout = BasicStringArrayLayout::Create(3);
  EXPECT_THAT(layout->GetConcrete(), IsOkAndHolds(layout));
}

TEST(BasicStringArrayLayoutTest, GetIfCustom) {
  std::shared_ptr<Layout> layout = BasicStringArrayLayout::Create(3);
  EXPECT_THAT(layout->GetIfCustom(), layout);
}

TEST(BasicStringArrayLayoutTest, ByteBound) {
  EXPECT_THAT(BasicStringArrayLayout::Create(3)->ByteBound(
                  DType(DType::kString), Shape({3})),
              IsOkAndHolds(Pair(0, sizeof(absl::string_view) * 3)));

  EXPECT_THAT(BasicStringArrayLayout::Create(3)->ByteBound(DType(DType::kS32),
                                                           Shape({3})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("BasicStringArrayLayout expects "
                                 "DType::kString, but got dtype=")));
  EXPECT_THAT(
      BasicStringArrayLayout::Create(3)->ByteBound(DType(DType::kString),
                                                   Shape({})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("BasicStringArrayLayout expects Shape with the "
                         "same number of elements, but got shard_shape=")));
}

TEST(BasicStringArrayLayoutTest, Equality) {
  EXPECT_EQ(*BasicStringArrayLayout::Create(2),
            *BasicStringArrayLayout::Create(2));

  EXPECT_NE(*BasicStringArrayLayout::Create(2),
            *BasicStringArrayLayout::Create(4));
  EXPECT_NE(*BasicStringArrayLayout::Create(2),
            *ByteStridesLayout::Create({2}));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
