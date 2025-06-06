/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_ENVIRONMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_ENVIRONMENT_H_

#include "cl_command_queue.h"
#include "cl_context.h"
#include "cl_device.h"
#include "program_cache.h"
#include "data_type.h"
#include "gpu_info.h"
#include "precision.h"
#include "status.h"
#include "tensor_desc.h"
#include "tensor.h"

namespace tflite {
namespace gpu {
namespace cl {

class Environment {
 public:
  Environment() = default;
  explicit Environment(CLDevice&& device, CLContext&& context,
                       CLCommandQueue&& queue,
                       ProfilingCommandQueue&& profiling_queue);
  // Move only
  Environment(Environment&& environment);
  Environment& operator=(Environment&& environment);
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;

  const CLDevice& device() const { return device_; }
  CLDevice* GetDevicePtr() { return &device_; }
  const CLDevice* GetDevicePtr() const { return &device_; }
  CLContext& context() { return context_; }
  CLCommandQueue* queue() { return &queue_; }
  ProfilingCommandQueue* profiling_queue() { return &profiling_queue_; }
  ProgramCache* program_cache() { return &program_cache_; }
  const ProgramCache* program_cache() const { return &program_cache_; }

  std::vector<CalculationsPrecision> GetSupportedPrecisions() const;
  bool IsSupported(CalculationsPrecision precision) const;
  std::vector<TensorStorageType> GetSupportedStorages() const;
  // returns storage types that support zero clamping when reading OOB in HW
  // (Height/Width) dimensions.
  std::vector<TensorStorageType> GetSupportedStoragesWithHWZeroClampSupport()
      const;
  bool IsSupported(TensorStorageType storage_type) const;

  absl::Status Init();

  void SetHighPerformance() const;
  void SetDefaultPerformance() const;
  void SetLowPerformance() const;  // for energy saving

 private:
  CLDevice device_;
  CLContext context_;
  CLCommandQueue queue_;
  ProfilingCommandQueue profiling_queue_;
  ProgramCache program_cache_;
};

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info);
TensorStorageType GetStorageTypeWithMinimalMemoryConsumption(
    const GpuInfo& gpu_info);

// Checks if image 2D creation from sub-buffer is supported.
bool CanUseSubBufferForImage2d(const GpuInfo& gpu_info);

absl::Status CreateEnvironment(Environment* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_ENVIRONMENT_H_
