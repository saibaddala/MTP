/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVERSION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVERSION_H_

#include "gpu_operation.h"

namespace tflite {
namespace gpu {

GPUOperation CreateTensorToTensorOp(const GpuInfo& gpu_info,
                                    const TensorDescriptor& src_desc,
                                    const TensorDescriptor& dst_desc);

GPUOperation CreateTensorToBhwcBufferOp(const GpuInfo& gpu_info,
                                        const TensorDescriptor& src_desc,
                                        const BufferDescriptor& dst_desc);

GPUOperation CreateBhwcBufferToTensorOp(const GpuInfo& gpu_info,
                                        const BufferDescriptor& src_desc,
                                        const TensorDescriptor& dst_desc);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVERSION_H_
