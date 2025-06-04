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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_TF_QUANT_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_TF_QUANT_OPS_H_

#include "Traits.h"  // from @llvm-project
#include "Attributes.h"  // from @llvm-project
#include "Builders.h"  // from @llvm-project
#include "BuiltinOps.h"  // from @llvm-project
#include "BuiltinTypes.h"  // from @llvm-project
#include "Dialect.h"  // from @llvm-project
#include "Matchers.h"  // from @llvm-project
#include "OpImplementation.h"  // from @llvm-project
#include "TypeUtilities.h"  // from @llvm-project
#include "CallInterfaces.h"  // from @llvm-project
#include "ControlFlowInterfaces.h"  // from @llvm-project
#include "DerivedAttributeOpInterface.h"  // from @llvm-project
#include "InferTypeOpInterface.h"  // from @llvm-project
#include "LoopLikeInterface.h"  // from @llvm-project
#include "SideEffectInterfaces.h"  // from @llvm-project
#include "tf_attributes.h"
#include "tf_op_interfaces.h"
#include "tf_structs.h"
#include "tf_traits.h"
#include "tf_types.h"
#include "tf_verifiers.h"

#define GET_OP_CLASSES
#include "tf_quant_ops.h.inc"

namespace mlir {
namespace quant {

// Function to register TensorFlow Uniform Quantized ops.
void RegisterOps();

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_TF_QUANT_OPS_H_
