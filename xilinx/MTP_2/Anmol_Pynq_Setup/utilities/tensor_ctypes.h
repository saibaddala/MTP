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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_CTYPES_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_CTYPES_H_

#include <vector>

#include "common.h"
#include "macros.h"
#include "types.h"

namespace tflite {

template <typename T>
inline T* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? reinterpret_cast<T*>(tensor->data.raw) : nullptr;
}

template <typename T>
inline const T* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr ? reinterpret_cast<const T*>(tensor->data.raw)
                           : nullptr;
}

TFLITE_NOINLINE RuntimeShape GetTensorShape(const TfLiteTensor* tensor);
RuntimeShape GetTensorShape(std::vector<int32_t> data);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_CTYPES_H_
