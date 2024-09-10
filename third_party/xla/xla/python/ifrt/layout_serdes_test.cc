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

#include <gtest/gtest.h>
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/layout_test_util.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

class LayoutSerDesTest : public test_util::LayoutTest {};

TEST_P(LayoutSerDesTest, DefaultLayoutRoundTrip) {
  auto layout = DefaultLayout::Create(DType(DType::kF32), Shape({3, 2}),
                                      /*device=*/client()->devices().front(),
                                      /*memory_kind=*/MemoryKind("abc"));

  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*layout, /*options=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_layout,
      Deserialize<DefaultLayout>(
          serialized, std::make_unique<DeserializeLayoutOptions>(client())));

  EXPECT_EQ(*out_layout, *layout);
}

TEST_P(LayoutSerDesTest, ByteStridesLayoutRoundTrip) {
  auto layout = ByteStridesLayout::Create({3, 2});

  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*layout, /*options=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_layout,
      Deserialize<ByteStridesLayout>(
          serialized, std::make_unique<DeserializeLayoutOptions>(client())));

  EXPECT_EQ(*out_layout, *layout);
}

INSTANTIATE_TEST_SUITE_P(NumDevices, LayoutSerDesTest,
                         testing::Values(test_util::LayoutTestParam{
                             /*num_devices=*/2}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
