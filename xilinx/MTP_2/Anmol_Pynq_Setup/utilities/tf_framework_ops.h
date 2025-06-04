/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the operations used in the TFFramework dialect.
//
#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_IR_TF_FRAMEWORK_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_IR_TF_FRAMEWORK_OPS_H_

#include "status.h"
#include "AllocationOpInterface.h"  // from @llvm-project
#include "Attributes.h"  // from @llvm-project
#include "BuiltinTypes.h"  // from @llvm-project
#include "Dialect.h"  // from @llvm-project
#include "MLIRContext.h"  // from @llvm-project
#include "OpDefinition.h"  // from @llvm-project
#include "OpImplementation.h"  // from @llvm-project
#include "ControlFlowInterfaces.h"  // from @llvm-project
#include "SideEffectInterfaces.h"  // from @llvm-project
#include "tf_status.h.inc"
#include "error_codes.pb.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

/// OpKernelContextType corresponds to C++ class OpKernelContext defined in
/// tensorflow/core/framework/op_kernel.h
class OpKernelContextType
    : public Type::TypeBase<OpKernelContextType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class JITCallableType
    : public Type::TypeBase<JITCallableType, Type, TypeStorage> {
 public:
  using Base::Base;
};

absl::StatusCode ConvertAttrToEnumValue(ErrorCode error_code);

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#define GET_OP_CLASSES
#include "tf_framework_dialect.h.inc"
#include "tf_framework_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_IR_TF_FRAMEWORK_OPS_H_
