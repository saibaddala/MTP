/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_

#include <memory>
#include <vector>

#include "hlo_module.h"
#include "gpu_device_info.h"

namespace xla {
namespace gpu {

int64_t GetSizeOfShape(const Shape& shape, int pointer_size);

// Determines the schedule of HLO instructions for a module run on the GPU.
Status ScheduleGpuModule(HloModule* module, int64_t pointer_size,
                         const GpuDeviceInfo& gpu_info);
HloInstructionSequence PostProcessSchedule(const HloInstructionSequence& input);

constexpr absl::string_view kFingerprintBeforeLHS = "fingerprint_before_lhs";

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_HLO_SCHEDULE_H_
