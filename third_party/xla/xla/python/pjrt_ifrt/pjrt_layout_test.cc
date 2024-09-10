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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::Pair;
using ::tsl::testing::IsOkAndHolds;

TEST(PjRtLayoutTest, Create) {
  EXPECT_FALSE(PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                                      xla::LayoutUtil::MakeDescendingLayout(1)))
                   ->is_default());
  EXPECT_EQ(PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                                   xla::LayoutUtil::MakeDescendingLayout(1)))
                ->pjrt_layout()
                ->xla_layout(),
            xla::LayoutUtil::MakeDescendingLayout(1));
}

TEST(PjRtLayoutTest, GetConcrete) {
  std::shared_ptr<Layout> layout =
      PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
          xla::LayoutUtil::MakeDescendingLayout(1)));
  EXPECT_THAT(layout->GetConcrete(), IsOkAndHolds(layout));
}

TEST(PjRtLayoutTest, GetIfCustom) {
  std::shared_ptr<Layout> layout =
      PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
          xla::LayoutUtil::MakeDescendingLayout(1)));
  EXPECT_THAT(layout->GetIfCustom(), layout);
}

TEST(PjRtLayoutTest, ByteBound) {
  EXPECT_THAT(PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                                     xla::LayoutUtil::MakeDescendingLayout(0)))
                  ->ByteBound(DType(DType::kS32), Shape({})),
              IsOkAndHolds(Pair(0, 4)));
  EXPECT_THAT(PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                                     xla::LayoutUtil::MakeDescendingLayout(1)))
                  ->ByteBound(DType(DType::kS32), Shape({3})),
              IsOkAndHolds(Pair(0, 12)));
  EXPECT_THAT(PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(xla::Layout(
                                     /*minor_to_major=*/{1, 0},
                                     /*tiles=*/{xla::Tile({2, 128})},
                                     /*element_size_in_bits=*/32)))
                  ->ByteBound(DType(DType::kS32), Shape({1, 127})),
              IsOkAndHolds(Pair(0, 4 * (2 * 128))));
}

TEST(PjRtLayoutTest, Equality) {
  EXPECT_EQ(*PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                xla::LayoutUtil::MakeDescendingLayout(1))),
            *PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                xla::LayoutUtil::MakeDescendingLayout(1))));

  EXPECT_NE(*PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                xla::LayoutUtil::MakeDescendingLayout(1))),
            *PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                xla::LayoutUtil::MakeDescendingLayout(2))));

  EXPECT_NE(*PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
                xla::LayoutUtil::MakeDescendingLayout(1))),
            *ByteStridesLayout::Create({2}));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
