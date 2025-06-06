/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_TRANSFORMS_SPMD_EXPANDERS_NOOP_IFRT_SPMD_EXPANDER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_TRANSFORMS_SPMD_EXPANDERS_NOOP_IFRT_SPMD_EXPANDER_H_

#include "DenseMap.h"
#include "Operation.h"  // from @llvm-project
#include "Visitors.h"  // from @llvm-project
#include "LogicalResult.h"  // from @llvm-project
#include "ifrt_interfaces.h"
#include "sharding_param.h"

namespace xla::ifrt {

// SPMD expander for operations that does not requires actual expansion.
template <typename OpT>
class NoOpIfrtSpmdExpander
    : public xla::ifrt::IfrtSpmdExpandable::ExternalModel<
          NoOpIfrtSpmdExpander<OpT>, OpT> {
 public:
  mlir::FailureOr<mlir::Operation*> SpmdExpand(mlir::Operation* op) const {
    return op;
  }

  mlir::FailureOr<llvm::DenseMap<int, ShardingParam>> ComputeShardingForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, ShardingParam>& input_shardings) const {
    // TODO(b/261623129): implement this method when sharding propagation pass
    // is implemented.
    op->emitOpError(
        "Interface method `ComputeShardingForward` not implemented.");
    return mlir::failure();
  }

  mlir::FailureOr<llvm::DenseMap<int, ShardingParam>> ComputeShardingBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, ShardingParam>& output_shardings) const {
    // TODO(b/261623129): implement this method when sharding propagation pass
    // is implemented.
    op->emitOpError(
        "Interface method `ComputeShardingBackward` not implemented.");
    return mlir::failure();
  }
};

}  // namespace xla::ifrt

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_TRANSFORMS_SPMD_EXPANDERS_NOOP_IFRT_SPMD_EXPANDER_H_
