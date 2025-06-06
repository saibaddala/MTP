/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_HARDWARE_TYPE_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_HARDWARE_TYPE_UTILS_H_

#include "string_view.h"
#include "hardware_types.pb.h"

namespace tensorflow {
namespace profiler {

// Get peak single precision throughput of the GPU in GFLOPS per
// streaming multiprocessor.
double GetFlopMaxThroughputPerSM(const DeviceCapabilities& device_cap);

// Returns the GPU model name from the given DeviceCapabilities.
absl::string_view GpuModelName(const DeviceCapabilities& device_cap);

HardwareType ParseHardwareType(absl::string_view device_type);

// Returns true if the given hardware type has a device.
bool HasDevice(HardwareType x);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_HARDWARE_TYPE_UTILS_H_
