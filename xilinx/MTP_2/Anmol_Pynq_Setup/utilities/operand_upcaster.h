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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_OPERAND_UPCASTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_OPERAND_UPCASTER_H_

#include "hlo_module.h"
#include "op_expander_pass.h"

namespace xla {

// Inserts Convert to operands of instructions that allows result accumulation
// as wider integral types.
class OperandUpcaster : public OpExpanderPass {
 public:
  explicit OperandUpcaster(HloPredicate extra_filter = nullptr)
      : OpExpanderPass(std::move(extra_filter)) {}

  absl::string_view name() const override { return "operand_upcaster"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_OPERAND_UPCASTER_H_
