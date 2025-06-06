/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_

#include <limits>
#include <string>

#include "string_view.h"
#include "hlo_module.h"
#include "call_graph.h"
#include "hlo_pass_interface.h"

namespace xla {

// An HLO pass that replaces while loop conditionals to execute a known constant
// number of iterations and remove operations that are difficult to run in
// standalone tests, such as infeed/outfeed and collective operations.
class HloControlFlowFlattening : public HloModulePass {
 public:
  // While execution count specifies how many times the while loops in the
  // transformed graph will execute.
  // If remove_comm = true, remove all communication operations.
  // If remove_host_transfer = true, remove the host-transfer send and recv
  // operations.
  struct Options {
    int while_execution_count = 1;
    int max_outer_loop_count = std::numeric_limits<int>::max();
    int max_loop_count = std::numeric_limits<int>::max();
    bool remove_infeed_outfeed = true;
    bool flatten_while_loop = true;
    bool remove_comm = true;
    bool remove_host_transfer = false;
  };
  explicit HloControlFlowFlattening(const Options& options)
      : while_execution_count_(options.while_execution_count),
        max_outer_loop_count_(options.max_outer_loop_count),
        max_loop_count_(options.max_loop_count),
        remove_infeed_outfeed_(options.remove_infeed_outfeed),
        flatten_while_loop_(options.flatten_while_loop),
        remove_host_transfer_(options.remove_host_transfer),
        remove_comm_(options.remove_comm) {}
  ~HloControlFlowFlattening() override = default;
  absl::string_view name() const override { return "control-flow-flattening"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Replaces an infeed with a custom call.
  Status RemoveInfeed(HloInstruction* infeed_hlo) const;
  // Removes outfeeds and replaces the outfeed HLO with a side-effecting custom
  // call that ensures that XLA doesn't dead-code-eliminate the outfeeded values
  // but lowers to a no-op.
  Status RemoveOutfeed(HloInstruction* outfeed_hlo) const;
  // Flattens the while loop. Precondition: while_hlo is a while instruction.
  Status FlattenWhileLoop(HloInstruction* while_hlo,
                          const CallGraph& call_graph) const;
  // Replaces an id with a zero constant.
  Status RemoveId(HloInstruction* hlo) const;
  // Removes send and send-done with a custom call.
  Status RemoveSendDone(
      HloInstruction* send_done,
      absl::flat_hash_set<HloInstruction*>* additional_removed) const;
  // Removes recv and recv-done with a custom call.
  Status RemoveRecvDone(
      HloInstruction* recv_done,
      absl::flat_hash_set<HloInstruction*>* additional_removed) const;

  int while_execution_count_;
  int max_outer_loop_count_;
  int max_loop_count_;
  bool remove_infeed_outfeed_;
  bool flatten_while_loop_;
  bool remove_host_transfer_;

 protected:
  // Replaces a collective op with a custom call.
  Status RemoveCollective(HloInstruction* hlo) const;

  bool remove_comm_;
};

// Retrieves the original loop bound. If fail, return a default value. If bounds
// exceed a given max, returns the max. This function is more opportunistic than
// ComputeWhileLoopTripCount in the while loop analysis as it may return a
// constant found in a compare expression when it is not an actual bound.
int GetLoopBound(const HloInstruction& while_hlo, const int default_loop_count,
                 const int max_loop_count);

// Retrieves the loop bound determined by the original loop bound, the max
// outer loops count and max loop count.
int GetLoopBoundWithOuterLoopMax(const HloInstruction& while_hlo,
                                 const CallGraph& call_graph,
                                 const int default_loop_count,
                                 const int max_outer_loop_count,
                                 const int max_loop_count);
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_
