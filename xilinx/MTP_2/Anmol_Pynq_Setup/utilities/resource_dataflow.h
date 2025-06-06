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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_RESOURCE_DATAFLOW_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_RESOURCE_DATAFLOW_H_

#include <algorithm>
#include <vector>

#include "BitVector.h"
#include "STLExtras.h"
#include "Casting.h"
#include "Debug.h"
#include "DeadCodeAnalysis.h"  // from @llvm-project
#include "SparseAnalysis.h"  // from @llvm-project
#include "FuncOps.h"  // from @llvm-project
#include "Builders.h"  // from @llvm-project
#include "BuiltinOps.h"  // from @llvm-project
#include "SymbolTable.h"  // from @llvm-project
#include "Value.h"  // from @llvm-project
#include "Pass.h"  // from @llvm-project
#include "LLVM.h"  // from @llvm-project

namespace mlir {
namespace TF {

// Used as a lattice value.
struct ResourceConstructingOps {
  explicit ResourceConstructingOps(Operation *op = nullptr);
  static ResourceConstructingOps getPessimisticValueState(MLIRContext *context);
  static ResourceConstructingOps getPessimisticValueState(Value value);
  bool operator==(const ResourceConstructingOps &rhs) const {
    return ops == rhs.ops;
  }

  static ResourceConstructingOps join(const ResourceConstructingOps &lhs,
                                      const ResourceConstructingOps &rhs);
  void print(raw_ostream &os) const;

  // The operation(s) which created the resource value.
  // IR constructs (i.e., GlobalTensorOp) are not const-correct.
  mutable DenseSet<Operation *> ops;
};

class ResourceDataflowAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          dataflow::Lattice<ResourceConstructingOps>> {
 public:
  using StateT = dataflow::Lattice<ResourceConstructingOps>;
  using dataflow::SparseForwardDataFlowAnalysis<
      StateT>::SparseForwardDataFlowAnalysis;
  ~ResourceDataflowAnalysis() override = default;

  void visitOperation(Operation *op, ArrayRef<const StateT *> operands,
                      ArrayRef<StateT *> results) override;

  void setToEntryState(StateT *lattice) override {
    propagateIfChanged(
        lattice,
        lattice->join(ResourceConstructingOps::getPessimisticValueState(
            lattice->getPoint())));
  }
};

}  // namespace TF
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_RESOURCE_DATAFLOW_H_
