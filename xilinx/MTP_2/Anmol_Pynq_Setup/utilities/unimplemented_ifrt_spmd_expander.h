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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_TRANSFORMS_SPMD_EXPANDERS_UNIMPLEMENTED_IFRT_SPMD_EXPANDER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_TRANSFORMS_SPMD_EXPANDERS_UNIMPLEMENTED_IFRT_SPMD_EXPANDER_H_

#include "DenseMap.h"
#include "Operation.h"  // from @llvm-project
#include "LogicalResult.h"  // from @llvm-project
#include "ifrt_interfaces.h"
#include "sharding_param.h"

namespace xla {
namespace ifrt {

// A temporary placeholder used for operations whose SPMD expanders have not
// been implemented. Using this class as operation's SPMD expander will suppress
// error from `SpmdExpandableInterfaceVerificationPass`.The usage of this class
// should be temporary and is generally discouraged as it will delay the error
// in the pipeline.
template <typename OpT>
class UnimplementedIfrtSpmdExpander
    : public xla::ifrt::IfrtSpmdExpandable::ExternalModel<
          UnimplementedIfrtSpmdExpander<OpT>, OpT> {
 public:
  mlir::FailureOr<mlir::Operation*> SpmdExpand(mlir::Operation* op) const {
    op->emitOpError("Interface method `SpmdExpand` not implemented.");
    return mlir::failure();
  }

  mlir::FailureOr<llvm::DenseMap<int, ShardingParam>> ComputeShardingForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, ShardingParam>& input_shardings) const {
    op->emitOpError(
        "Interface method `ComputeShardingForward` not implemented.");
    return mlir::failure();
  }

  mlir::FailureOr<llvm::DenseMap<int, ShardingParam>> ComputeShardingBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, ShardingParam>& output_shardings) const {
    op->emitOpError(
        "Interface method `ComputeShardingBackward` not implemented.");
    return mlir::failure();
  }
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_TRANSFORMS_SPMD_EXPANDERS_UNIMPLEMENTED_IFRT_SPMD_EXPANDER_H_
