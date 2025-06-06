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

#ifndef TENSORFLOW_TSL_FRAMEWORK_FIXEDPOINT_FIXEDPOINT_H_
#define TENSORFLOW_TSL_FRAMEWORK_FIXEDPOINT_FIXEDPOINT_H_

#include "Tensor"
#include "fixedpoint_types.h"

// Use optimized implementations whenever available
#if defined(EIGEN_VECTORIZE_AVX512DQ) || defined(EIGEN_VECTORIZE_AVX512BW)
#include "PacketMathAVX512.h"
#include "TypeCastingAVX512.h"

#elif defined EIGEN_VECTORIZE_AVX2
#define EIGEN_USE_OPTIMIZED_INT8_UINT8_MAT_MAT_PRODUCT
#define EIGEN_USE_OPTIMIZED_INT16_INT16_MAT_MAT_PRODUCT
#include "PacketMathAVX2.h"
// Disable clang-format to prevent 'MatMatProductAVX2.h' header from being
// included before 'PacketMathAVX2' header on which it depends.
// clang-format off
#include "MatMatProductAVX2.h"
// clang-format on
#include "TypeCastingAVX2.h"

#elif defined EIGEN_VECTORIZE_AVX
#include "PacketMathAVX.h"

#elif defined EIGEN_VECTORIZE_NEON
#define EIGEN_USE_OPTIMIZED_INT8_INT8_MAT_MAT_PRODUCT
#define EIGEN_USE_OPTIMIZED_INT8_UINT8_MAT_MAT_PRODUCT
#define EIGEN_USE_OPTIMIZED_UINT8_INT8_MAT_MAT_PRODUCT
#define EIGEN_USE_OPTIMIZED_INT16_INT16_MAT_MAT_PRODUCT
#include "MatMatProductNEON.h"
#endif

// Use the default implementation when no optimized code is available
#include "MatMatProduct.h"
#include "MatVecProduct.h"

#endif  // TENSORFLOW_TSL_FRAMEWORK_FIXEDPOINT_FIXEDPOINT_H_
