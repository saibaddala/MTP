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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_DW_CONVOLUTION_SELECTOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_DW_CONVOLUTION_SELECTOR_H_

#include <memory>

#include "operations.h"
#include "status.h"
#include "gpu_operation.h"

namespace tflite {
namespace gpu {

std::unique_ptr<GPUOperation> SelectDWConvolution(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_DW_CONVOLUTION_SELECTOR_H_
