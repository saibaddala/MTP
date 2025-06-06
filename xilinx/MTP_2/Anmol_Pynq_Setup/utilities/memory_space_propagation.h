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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_PROPAGATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_PROPAGATION_H_

#include "hlo_module.h"
#include "hlo_dataflow_analysis.h"
#include "hlo_pass_interface.h"

namespace xla {

// This is a legalization pass that propagates the memory space in the layout to
// the fusion computations.
class MemorySpacePropagation : public HloModulePass {
 public:
  ~MemorySpacePropagation() override = default;
  absl::string_view name() const override { return "memory-space-propagation"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Given the shape index (operand or output) and its corresponding instruction
  // in the fused computation (parameter or root), propagates the memory space
  // in the callee side. Returns true if the module is modified.
  bool Propagate(ShapeIndexView index, const HloInstruction* callee_instruction,
                 int64_t memory_space) const;

  std::unique_ptr<HloDataflowAnalysis> dataflow_analysis_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_PROPAGATION_H_
