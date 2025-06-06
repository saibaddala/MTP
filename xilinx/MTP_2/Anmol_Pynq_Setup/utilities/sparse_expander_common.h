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

#ifndef TENSORFLOW_DTENSOR_MLIR_SPARSE_EXPANDER_COMMON_H_
#define TENSORFLOW_DTENSOR_MLIR_SPARSE_EXPANDER_COMMON_H_

#include <optional>

#include "optional.h"
#include "Operation.h"  // from @llvm-project
#include "tf_ops.h"

namespace tensorflow {
namespace dtensor {

// Gets the SparseToDenseOp that generates `value` if `value` is the result of
// a SparseToDenseOp. Returns empty otherwise. This is useful
// in SparseExpansion where we want to check whether some operand
// is a SparseTensor, by checking whether that operand is a result of a
// SparseToDenseOp. If this value is eventually an output of a SparseToDenseOp,
// there should only be DTensor related ops between the actual SparseToDenseOp,
// e.g. DTensorRelayout ops or DTensorLayout op.
StatusOr<mlir::TF::SparseToDenseOp> GetSparseToDenseOp(mlir::Value value);

// Checks whether `value is an output of a SparseToDenseOp value.
bool IsSparseValue(mlir::Value value);

// Checks if `op` has any sparse value operands.
bool HasAnySparseInput(mlir::Operation* op);

// Checks if all operands of `op` is a sparse value.
bool AllSparseInput(mlir::Operation* op);

// Returns the indices component dense tensor from `value`. `value` represents
// a SparseTensor value.
StatusOr<mlir::Value> GetIndicesFromSparseTensor(mlir::Value value);

// Returns the values component dense tensor from `value`.`value` represents
// a SparseTensor value.
StatusOr<mlir::Value> GetValuesFromSparseTensor(mlir::Value value);

// Returns the dense shape component dense tensor from `value`. `value`
// represents a SparseTensor value.
StatusOr<mlir::Value> GetDenseShapesFromSparseTensor(mlir::Value value);

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_SPARSE_EXPANDER_COMMON_H_
