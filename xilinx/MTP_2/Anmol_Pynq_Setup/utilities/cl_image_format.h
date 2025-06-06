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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_IMAGE_FORMAT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_IMAGE_FORMAT_H_

#include "opencl_wrapper.h"
#include "data_type.h"

namespace tflite {
namespace gpu {
namespace cl {

cl_channel_order ToChannelOrder(int num_channels);

cl_channel_type DataTypeToChannelType(DataType type, bool normalized = false);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_IMAGE_FORMAT_H_
