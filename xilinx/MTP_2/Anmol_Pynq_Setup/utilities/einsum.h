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
// This pass identifies patterns for certain Einsum Ops and replaces them
// with other equivalent TF Ops.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_EINSUM_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_EINSUM_H_

#include <cstdint>
#include <initializer_list>

#include "ArrayRef.h"
#include "Casting.h"
#include "Attributes.h"  // from @llvm-project
#include "BuiltinTypes.h"  // from @llvm-project
#include "Location.h"  // from @llvm-project
#include "MLIRContext.h"  // from @llvm-project
#include "Matchers.h"  // from @llvm-project
#include "PatternMatch.h"  // from @llvm-project
#include "TypeUtilities.h"  // from @llvm-project
#include "Pass.h"  // from @llvm-project
#include "tf_ops.h"
#include "matmul_bcast.h"

namespace mlir {
namespace TF {

// TF.Einsum provides fully general tensor contractions. For a few select
// cases, we can convert this op to other TF Ops, which in later passes
// properly convert to TF Lite ops.
struct ConvertTFEinsumOp : public OpRewritePattern<TF::EinsumOp> {
 public:
  explicit ConvertTFEinsumOp(MLIRContext* context)
      : OpRewritePattern<TF::EinsumOp>(context) {}

  LogicalResult matchAndRewrite(TF::EinsumOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_EINSUM_H_
