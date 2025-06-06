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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_UTILS_HLO_QUERY_H_
#define TENSORFLOW_COMPILER_XLA_HLO_UTILS_HLO_QUERY_H_

#include "flat_hash_set.h"
#include "hlo_computation.h"
#include "hlo_instruction.h"
#include "hlo_module.h"
#include "hlo_opcode.h"

namespace xla {

// Helper interface for making queries about the HLO IR.
namespace hlo_query {

// Returns whether the given opcode is a collective communications operation
// that is represented as HloCollectiveInstruction.
bool IsCollectiveCommunicationOp(HloOpcode op);

// Returns whether the given opcode represents the start operation for a
// collective communication, may include send & recv operations.
bool IsAsyncCollectiveStartOp(HloOpcode op, bool include_send_recv = false);
// Returns whether the given opcode represents the done operation for a
// collective communication, may include send & recv operations.
bool IsAsyncCollectiveDoneOp(HloOpcode op, bool include_send_recv = false);

// Returns whether the instruction provided is a constant rank-0 float32, and
// if so, places the constant value into out.
// Precondition: out != nullptr
bool IsConstantR0F32(HloInstruction* instruction, float* out);

// Returns whether all of an instruction's operands are of the types constants
// and parameters.
bool AllOperandsAreParametersOrConstants(const HloInstruction& instruction);

// Returns whether all of an instruction's operands are parameters.
bool AllOperandsAreParameters(const HloInstruction& instruction);

// Returns whether all of an instruction's operands are constants.
bool AllOperandsAreConstants(const HloInstruction& instruction);

// Returns whether the instruction is a scalar constant.
bool IsScalarConstant(const HloInstruction* instruction);

// Returns whether the `instr` is either a constant, a scalar, or a
// broadcasted constant/scalar.
bool IsBroadcastedConstantOrScalar(const HloInstruction& instr);

// Returns whether the `instr` is a broadcast and its input is a
// scalar constant.
bool IsBroadcastOfScalarConstant(const HloInstruction& instr);

// Returns first HLO of the computation with the opcode, otherwise nullptr.
HloInstruction* GetFirstInstructionWithOpcode(const HloComputation& computation,
                                              HloOpcode opcode);

// Determines whether the given computation contains an instruction with one of
// the given opcodes.  Checks both comp's instructions and the instructions of
// any computations nested within it.
bool ContainsInstrWithOpcode(const HloComputation* comp,
                             const absl::flat_hash_set<HloOpcode>& opcodes);

// Returns an operand of an instruction with the given opcode. If there are
// multiple matching operands, then the first matching operand is returned. If
// there are no matching operands then nullptr is returned.
HloInstruction* GetMatchingOperand(const HloPredicate& matcher,
                                   HloInstruction* instruction);

// Returns whether a binary instruction has a matching operand. Sets
// matching_operand to the matching operand and the other operand to
// other_operand. Note: in the case where both operands match, the first operand
// of the instruction is returned.
bool MatchBinaryInstructionOperand(const HloPredicate& matcher,
                                   HloInstruction* instruction,
                                   HloInstruction** matching_operand,
                                   HloInstruction** other_operand);

// Returns whether a binary instruction has a operand with a given opcode.
// This is a special case of MatchingBinaryInstructionOperand.
bool MatchBinaryInstructionOperandOpcode(HloOpcode opcode,
                                         HloInstruction* instruction,
                                         HloInstruction** matching_operand,
                                         HloInstruction** other_operand);

// Returns whether the module contains the given collective communication
// instructions with constrained layout.
bool ContainsLayoutConstrainedCollective(const HloModule& module, HloOpcode op);

// Returns whether the module contains all-reduce instructions with constrained
// layout.
inline bool ContainsLayoutConstrainedAllReduce(const HloModule& module) {
  return ContainsLayoutConstrainedCollective(module, HloOpcode::kAllReduce);
}

// Returns the next available channel id that can be used in the given module
// (for HloChannelInstructions).
int64_t NextChannelId(const HloModule& module);

// Returns whether the module contains host send/recv with X64 data type.
// This function is called after X64Rewriter, so X64 host transfers are already
// rewritten into tuple shaped transfers.
bool HasX64TransformedHostTransfer(const HloModule& module);

}  // namespace hlo_query
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_UTILS_HLO_QUERY_H_
