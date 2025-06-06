/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_MERGE_DENSIFY_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_MERGE_DENSIFY_H_

#include <memory>

#include "model_transformer.h"

namespace tflite {
namespace gpu {

// Merges DENSIFY with CONV_2D / DEPTHWISE_CONV_2D.
std::unique_ptr<NodeTransformation> NewMergeDensify();

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_MERGE_DENSIFY_H_
