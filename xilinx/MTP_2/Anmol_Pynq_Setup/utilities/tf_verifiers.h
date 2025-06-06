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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_VERIFIERS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_VERIFIERS_H_

#include "Operation.h"  // from @llvm-project
#include "LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace TF {

// Verifies correctness of ops implementing LayoutSensitiveInterface (see
// definition in tf_op_base.td):
// (1) Operation must have valid `data_format` attribute.
// (2) Layout dependent arguments and results indices must be in
//     [0, getNumOperands/getNumResults) range.
LogicalResult VerifyLayoutSensitiveInterface(Operation* op);

// Verifies correctness of ops implementing FoldOperandsTransposeInterface (see
// definition in tf_op_base.td):
// (1) Layout dependent arguments and results indices must be in
//     [0, getNumOperands/getNumResults) range.
LogicalResult VerifyFoldOperandsTransposeInterface(Operation* op);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_VERIFIERS_H_
