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

#ifndef MLIR_HLO_GML_ST_TRANSFORMS_PEELING_PEELING_H
#define MLIR_HLO_GML_ST_TRANSFORMS_PEELING_PEELING_H

#include "gml_st_ops.h"
#include "SCF.h"
#include "PatternMatch.h"

namespace mlir {
namespace gml_st {

struct GmlStPeelingResult {
  scf::ForallOp mainLoop = nullptr;
  SmallVector<scf::ForallOp> tailLoops = {};
};

/// Rewrite a scf::ForallOp with bounds/step that potentially do not divide
/// evenly into a scf::ForallOp where the step divides the iteration space
/// evenly, followed by another scf::ForallOp for the last (partial)
/// iteration (if any).  This transformation is called "loop peeling".
///
/// These functions peel all loops in the loop nest by calling
/// peelAndCanonicalizeGmlStLoop. Additionally, they mark all loops (main and
/// remainder loops) as peeled, so the same loop is not rewritten a second time.
GmlStPeelingResult peelAllLoops(scf::ForallOp loop,
                                mlir::PatternRewriter &rewriter);

struct SCFForPeelingResult {
  scf::ForOp mainLoop = nullptr;
  scf::ForOp tailLoop = nullptr;
};
SCFForPeelingResult peelSCFForOp(RewriterBase &rewriter, scf::ForOp);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_GML_ST_TRANSFORMS_PEELING_PEELING_H
