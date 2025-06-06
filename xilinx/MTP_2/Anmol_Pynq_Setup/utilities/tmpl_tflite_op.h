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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_TFLITE_OP_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_TFLITE_OP_H_

#include "common.h"
#include "mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace custom {

// Add TmplOp to the resolver
void AddTmplOp(MutableOpResolver* resolver);

// Creates and returns the op kernel
TfLiteRegistration* Register_TMPL_OP();

// The name of the op
const char* OpName_TMPL_OP();

}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_TFLITE_OP_H_
