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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_MLIR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_MLIR_EMITTER_H_

#include "IRBuilder.h"
#include "Value.h"
#include "FuncOps.h"  // from @llvm-project
#include "Builders.h"  // from @llvm-project
#include "BuiltinOps.h"  // from @llvm-project
#include "MLIRContext.h"  // from @llvm-project
#include "shape.h"
#include "status.h"

namespace xla {
namespace cpu {

// Create a new MLIR function with the name `func_name`, populate it with
// `emitter` and create a call, passing it the buffers defined by
// resultShape/resultPtr and operandShapes/operandPtrs. The function is added to
// the LLVM module at `b`s insertion point.
Status EmitMlirFuncAndCall(
    mlir::MLIRContext *context, llvm::IRBuilder<> *b, const Shape &result_shape,
    llvm::ArrayRef<Shape> operand_shapes, llvm::Value *result_ptr,
    llvm::ArrayRef<llvm::Value *> operand_ptrs, llvm::StringRef func_name,
    llvm::function_ref<void(mlir::OpBuilder *, mlir::func::FuncOp)> emitter);

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_MLIR_EMITTER_H_
