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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_INSTRUCTION_HOISTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_INSTRUCTION_HOISTER_H_

#include "hlo_module.h"
#include "hlo_pass_interface.h"

namespace xla {

// HLO pass that hoists parameters and constants to increase opportunities for
// prefetching.
class InstructionHoister : public HloModulePass {
 public:
  explicit InstructionHoister(bool hoist_parameters = true,
                              bool host_constants = true)
      : hoist_parameters_(hoist_parameters), host_constants_(host_constants) {}

  ~InstructionHoister() override = default;

  absl::string_view name() const override { return "instruction-hoister"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool hoist_parameters_;
  bool host_constants_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INSTRUCTION_HOISTER_H_
