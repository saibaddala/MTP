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

// This file defines the operations used in the MLIR TensorFlow Lite dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_H_

#include "Traits.h"  // from @llvm-project
#include "Attributes.h"  // from @llvm-project
#include "Builders.h"  // from @llvm-project
#include "BuiltinTypes.h"  // from @llvm-project
#include "Dialect.h"  // from @llvm-project
#include "DialectImplementation.h"  // from @llvm-project
#include "OpImplementation.h"  // from @llvm-project
#include "DerivedAttributeOpInterface.h"  // from @llvm-project
#include "InferTypeOpInterface.h"  // from @llvm-project
#include "LoopLikeInterface.h"  // from @llvm-project
#include "SideEffectInterfaces.h"  // from @llvm-project
#include "LLVM.h"  // from @llvm-project
#include "TypeID.h"  // from @llvm-project
#include "tfl_ops_dialect.h.inc"
#include "tfl_ops_enums.h.inc"
#include "QuantOps.h"
#include "quantization_utils.h"
#include "utils.h"
#include "tf_traits.h"
#include "schema_generated.h"
#define GET_ATTRDEF_CLASSES
#include "tfl_ops_attrdefs.h.inc"

namespace mlir {
namespace TFL {

typedef TFLDialect TensorFlowLiteDialect;

// The Control type is a token-like value that models control dependencies
class ControlType : public Type::TypeBase<ControlType, Type, TypeStorage> {
 public:
  using Base::Base;
};

#include "tfl_ops_interface.h.inc"

}  // end namespace TFL
}  // end namespace mlir

#define GET_OP_CLASSES
#include "tfl_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_H_
