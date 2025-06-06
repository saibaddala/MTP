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
#ifndef TENSORFLOW_C_EAGER_TFE_TENSOR_DEBUG_INFO_INTERNAL_H_
#define TENSORFLOW_C_EAGER_TFE_TENSOR_DEBUG_INFO_INTERNAL_H_

#include <vector>

#include "types.h"

struct TFE_TensorDebugInfo {
  explicit TFE_TensorDebugInfo(const std::vector<int64_t>& dims)
      : dev_dims(dims) {}

  // Fully-padded, minor-to-major.
  std::vector<int64_t> dev_dims;
};

#endif  // TENSORFLOW_C_EAGER_TFE_TENSOR_DEBUG_INFO_INTERNAL_H_
