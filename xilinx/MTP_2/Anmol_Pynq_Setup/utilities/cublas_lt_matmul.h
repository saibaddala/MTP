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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_CUBLAS_LT_MATMUL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_CUBLAS_LT_MATMUL_H_

#include "custom_call_encoding.h"
#include "custom_call_registry.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "matmul_utils.h"
#if GOOGLE_CUDA
#include "cuda_blas_lt.h"
#else
#include "rocm_config.h"
#include "hip_blas_lt.h"
#endif
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

// Registers XLA Gpu runtime cuBLASLt custom calls.
void RegisterMatmulCustomCalls(runtime::DirectCustomCallRegistry& registry);

// Add cuBLASLt attributes encoding
void PopulateCublasLtMatmulAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding);

#if GOOGLE_CUDA || TF_HIPBLASLT
// Keep cublas_lt::MatmulPlan's for all matmul instances in the executable.
class MatmulPlans : public runtime::StateVector<cublas_lt::MatmulPlan> {};
#endif  // GOOGLE_CUDA || TF_HIPBLASLT

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_CUBLAS_LT_MATMUL_H_
