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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSIONS_IN_PLACE_DYNAMIC_UPDATE_SLICE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSIONS_IN_PLACE_DYNAMIC_UPDATE_SLICE_H_

#include <vector>

#include "fusion_emitter.h"
#include "ir_emission_utils.h"

namespace xla {
namespace gpu {

// Fusion node where the root is either:
// 1. a dynamic-update-slice op
// 2. a bitcast of a dynamic-update-slice op
// 3. a tuple op returning the result of several dynamic-update-slice ops
// 4. a tuple op returning the result of several bitcast
//    dynamic-update-slice ops
//
// Additionally, all the dynamic-update-slice ops have exactly one user. The
// fusion parameter that they update can have users (in addition to the
// dynamic-update-slice op) that read in either
// a. a dynamic-slice corresponding exactly to the slice of the parameter that
//    is updated by the dynamic-update-slice op
// b. a dynamic-slice reading in a single element anywhere in the parameter.
//    This is only allowed if the dynamic-update-slice op updates a single
//    element
//
// In both cases, the additional users must not flow into any other output
// than the dynamic-slice-update corresponding to that particular slice of the
// parameter.
//
// The assumption is that each op's input (i.e. array to update) shares the
// same slice as its output. In this case, we have a special algorithm that
// modifies the output in place without touching the un-updated elements. The
// update slice is assumed to be the exact same for all the
// dynamic-update-slice ops.
class InPlaceDynamicUpdateSliceEmitter : public KernelFusionEmitterBase {
 public:
  InPlaceDynamicUpdateSliceEmitter(IrEmitterContext& ir_emitter_context,
                                   ElementalIrEmitter& elemental_emitter,
                                   mlir::lmhlo::FusionOp fusion_op,
                                   const HloFusionInstruction& fusion)
      : KernelFusionEmitterBase(ir_emitter_context, elemental_emitter,
                                fusion_op, fusion),
        dus_ops_(GetOutputDefiningDynamicUpdateSlices(
            fusion.fused_instructions_computation())) {}
  StatusOr<LaunchDimensions> launch_dimensions() const override;

 protected:
  Status EmitKernel(const LaunchDimensions& launch_dims,
                    std::vector<llvm_ir::IrArray> inputs,
                    std::vector<llvm_ir::IrArray> outputs,
                    llvm::IRBuilder<>* builder,
                    int kernel_index) const override;

  std::vector<HloInstruction*> dus_ops_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSIONS_IN_PLACE_DYNAMIC_UPDATE_SLICE_H_
