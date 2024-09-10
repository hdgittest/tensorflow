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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout_test_util.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::Pair;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

class DefaultLayoutTest : public test_util::LayoutTest {};

TEST_P(DefaultLayoutTest, Create) {
  auto layout =
      DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
                            client()->devices().front(), MemoryKind("host"));
  EXPECT_TRUE(layout->is_default());
  EXPECT_EQ(layout->dtype(), DType(DType::kS32));
  EXPECT_EQ(layout->shard_shape(), Shape({3, 2}));
  EXPECT_EQ(layout->device(), client()->devices().front());
  EXPECT_EQ(layout->memory_kind(), MemoryKind("host"));
}

TEST_P(DefaultLayoutTest, GetConcrete) {
  // TODO(hyeontaek): Remove skipTest() when implemented.
  GTEST_SKIP() << "Not implemented.";

  // auto layout =
  //     DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
  //                           client()->devices().front(), MemoryKind("host"));
  // EXPECT_THAT(layout->GetConcrete(), IsOkAndHolds(NotNull()));
}

TEST_P(DefaultLayoutTest, GetIfCustom) {
  auto layout =
      DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
                            client()->devices().front(), MemoryKind("host"));
  EXPECT_THAT(layout->GetIfCustom(), IsNull());
}

TEST_P(DefaultLayoutTest, Equality) {
  EXPECT_EQ(
      *DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
                             client()->devices().at(1), MemoryKind("host")),
      *DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
                             client()->devices().at(1), MemoryKind("host")));

  EXPECT_NE(
      *DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
                             client()->devices().at(1), MemoryKind("host")),
      *DefaultLayout::Create(DType(DType::kS8), Shape({3, 2}),
                             client()->devices().at(1), MemoryKind("host")));
  EXPECT_NE(
      *DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
                             client()->devices().at(1), MemoryKind("host")),
      *DefaultLayout::Create(DType(DType::kS32), Shape({3}),
                             client()->devices().at(1), MemoryKind("host")));
  EXPECT_NE(
      *DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
                             client()->devices().at(1), MemoryKind("host")),
      *DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
                             client()->devices().at(0), MemoryKind("host")));
  EXPECT_NE(
      *DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
                             client()->devices().at(1), MemoryKind("host")),
      *DefaultLayout::Create(DType(DType::kS32), Shape({3, 2}),
                             client()->devices().at(1), MemoryKind("device")));
}

class ByteStridesLayoutTest : public test_util::LayoutTest {};

TEST_P(ByteStridesLayoutTest, Create) {
  EXPECT_FALSE(ByteStridesLayout::Create({})->is_default());

  EXPECT_THAT(ByteStridesLayout::Create({})->byte_strides(), ElementsAre());
  EXPECT_THAT(ByteStridesLayout::Create({8, 4})->byte_strides(),
              ElementsAre(8, 4));
}

TEST_P(ByteStridesLayoutTest, CreateDenseMajorToMinor) {
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout,
                            ByteStridesLayout::CreateCompactMajorToMinor(
                                DType(DType::kToken), Shape({})));
    EXPECT_THAT(layout->byte_strides(), ElementsAre());
  }

  EXPECT_THAT(
      ByteStridesLayout::CreateCompactMajorToMinor(DType(DType::kToken),
                                                   Shape({3, 2})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "ByteStridesLayout expects DType that has a "
              "byte size if shape dimensions are not empty, but got dtype=")));

  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout,
                            ByteStridesLayout::CreateCompactMajorToMinor(
                                DType(DType::kS32), Shape({})));
    EXPECT_THAT(layout->byte_strides(), ElementsAre());
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto layout,
                            ByteStridesLayout::CreateCompactMajorToMinor(
                                DType(DType::kS32), Shape({3, 2})));
    EXPECT_THAT(layout->byte_strides(), ElementsAre(8, 4));
  }
}

TEST_P(ByteStridesLayoutTest, GetConcrete) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Layout> layout,
                          ByteStridesLayout::CreateCompactMajorToMinor(
                              DType(DType::kS32), Shape({})));
  EXPECT_THAT(layout->GetConcrete(), IsOkAndHolds(layout));
}

TEST_P(ByteStridesLayoutTest, GetIfCustom) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Layout> layout,
                          ByteStridesLayout::CreateCompactMajorToMinor(
                              DType(DType::kS32), Shape({})));
  EXPECT_THAT(layout->GetIfCustom(), layout);
}

TEST_P(ByteStridesLayoutTest, ByteBound) {
  EXPECT_THAT(
      ByteStridesLayout::Create({})->ByteBound(DType(DType::kToken), Shape({})),
      IsOkAndHolds(Pair(0, 0)));
  EXPECT_THAT(
      ByteStridesLayout::Create({})->ByteBound(DType(DType::kS8), Shape({})),
      IsOkAndHolds(Pair(0, 1)));
  EXPECT_THAT(
      ByteStridesLayout::Create({})->ByteBound(DType(DType::kS32), Shape({})),
      IsOkAndHolds(Pair(0, 4)));
  EXPECT_THAT(ByteStridesLayout::Create({8, 4})->ByteBound(DType(DType::kS32),
                                                           Shape({3, 2})),
              IsOkAndHolds(Pair(0, 24)));
  EXPECT_THAT(ByteStridesLayout::Create({-8, 4})->ByteBound(DType(DType::kS32),
                                                            Shape({3, 2})),
              IsOkAndHolds(Pair(-16, 8)));

  EXPECT_THAT(
      ByteStridesLayout::Create({})->ByteBound(DType(DType::kS32),
                                               Shape({3, 2})),
      StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("ByteStridesLayout expects Shape with the same number of "
                    "dimensions, but got shard_shape=")));
}

TEST_P(ByteStridesLayoutTest, Equality) {
  EXPECT_EQ(*ByteStridesLayout::Create({1}), *ByteStridesLayout::Create({1}));

  EXPECT_NE(*ByteStridesLayout::Create({1}), *ByteStridesLayout::Create({2}));
  EXPECT_NE(
      *ByteStridesLayout::Create({1}),
      *DefaultLayout::Create(DType(DType::kS32), Shape({1}),
                             client()->devices().at(0), MemoryKind("host")));
}

INSTANTIATE_TEST_SUITE_P(NumDevices, DefaultLayoutTest,
                         testing::Values(test_util::LayoutTestParam{
                             /*num_devices=*/2}));
INSTANTIATE_TEST_SUITE_P(NumDevices, ByteStridesLayoutTest,
                         testing::Values(test_util::LayoutTestParam{
                             /*num_devices=*/1}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
