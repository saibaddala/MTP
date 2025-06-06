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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_CONSTANT_FOLD_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_CONSTANT_FOLD_H_

#include "SmallVector.h"
#include "Operation.h"  // from @llvm-project
#include "PatternMatch.h"  // from @llvm-project

namespace mlir {
namespace quant {

// Applies constant folding recursively if the operation and all of its operands
// are foldable. Returns the constants generated by constant-folding or the
// original operation's outputs if not folded.
SmallVector<Value> ConstantFoldOpIfPossible(Operation* op);

// This pattern tries to constant-fold the quantizable operands of supported
// TF operations.
struct ConstantFoldQuantizableOperands : public RewritePattern {
 public:
  explicit ConstantFoldQuantizableOperands(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CC_CONSTANT_FOLD_H_
