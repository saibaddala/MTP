/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_PADDING_LEGALIZATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_PADDING_LEGALIZATION_H_

#include "hlo_pass_interface.h"

namespace xla {
namespace gpu {

// An HLO pass that canonicalizes convolution instructions for GPU codegen. It
// inserts Pad instructions before Convolution instructions with uncanonicalized
// padding, so that they can be lowered to Cudnn/Miopen convolution.
class GpuConvPaddingLegalization : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "gpu-conv-padding-legalization";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  StatusOr<bool> RunOnComputation(HloComputation* computation);
  // Returns if any changes are made to the parent computation.
  bool CanonicalizeForwardConvolution(HloInstruction* conv);
  bool CanonicalizeBackwardFilterConvolution(HloInstruction* backward_conv);
  bool CanonicalizeBackwardInputConvolution(HloInstruction* backward_conv);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_PADDING_LEGALIZATION_H_
