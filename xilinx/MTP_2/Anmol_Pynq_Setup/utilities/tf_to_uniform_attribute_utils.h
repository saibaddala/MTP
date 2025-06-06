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
// This header file defines common utils used when transforming TF ops to
// Uniform Quantized ops.

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_TF_TO_UNIFORM_ATTRIBUTE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_TF_TO_UNIFORM_ATTRIBUTE_UTILS_H_

#include "StringMap.h"
#include "Attributes.h"  // from @llvm-project
#include "PatternMatch.h"  // from @llvm-project
#include "utils.h"

namespace mlir::quant {

LogicalResult FillAttributesForUniformQuantizedDotOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    tensorflow::quantization::QuantizationMethod::ExperimentalMethod
        quantization_method,
    bool enable_per_channel_quantization);

LogicalResult FillAttributesForUniformQuantizedConvolutionOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    tensorflow::quantization::QuantizationMethod::ExperimentalMethod
        quantization_method,
    bool enable_per_channel_quantization);

LogicalResult FillAttributesForUniformQuantizedAddOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    tensorflow::quantization::QuantizationMethod::ExperimentalMethod
        quantization_method,
    bool enable_per_channel_quantization);

LogicalResult FillAttributesForUniformQuantizedClipByValueOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    tensorflow::quantization::QuantizationMethod::ExperimentalMethod
        quantization_method,
    bool enable_per_channel_quantization);

LogicalResult FillAttributesForUniformRequantizeOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    tensorflow::quantization::QuantizationMethod::ExperimentalMethod
        quantization_method,
    bool enable_per_channel_quantization);

}  // namespace mlir::quant

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_TF_TO_UNIFORM_ATTRIBUTE_UTILS_H_
