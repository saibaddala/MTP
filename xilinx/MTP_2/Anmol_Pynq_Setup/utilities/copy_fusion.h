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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COPY_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COPY_FUSION_H_

#include "hlo_module.h"
#include "hlo_pass_interface.h"
#include "statusor.h"

namespace xla {
namespace gpu {

// CopyFusion checks if a fusion is followed by multiple copies and if so, adds
// those copies to the fusion, replacing the copies with get_tuple_elements.
class CopyFusion : public HloModulePass {
 public:
  CopyFusion() = default;

  absl::string_view name() const override { return "copy_fusion"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  StatusOr<bool> DoCopyFusion(HloComputation* computation);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COPY_FUSION_H_
