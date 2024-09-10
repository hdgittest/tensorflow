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

#ifndef XLA_PYTHON_IFRT_LAYOUT_TEST_UTIL_H_
#define XLA_PYTHON_IFRT_LAYOUT_TEST_UTIL_H_

#include <memory>

#include "absl/types/span.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace test_util {

// Parameters for LayoutTest.
// Requests `num_devices` total devices.
struct LayoutTestParam {
  int num_devices;
};

// Test fixture for layout tests.
class LayoutTest : public testing::TestWithParam<LayoutTestParam> {
 public:
  void SetUp() override;
  Client* client() { return client_.get(); }

  // Returns `DeviceList` containing devices at given indexes (not ids) within
  // `client.devices()`.
  // REQUIRES: 0 <= device_indices[i] < num_devices
  DeviceListRef GetDevices(absl::Span<const int> device_indices);

 private:
  std::shared_ptr<Client> client_;
};

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_LAYOUT_TEST_UTIL_H_
