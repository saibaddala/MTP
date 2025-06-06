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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_IR_DFS_HLO_VISITOR_H_
#define TENSORFLOW_COMPILER_XLA_HLO_IR_DFS_HLO_VISITOR_H_

#include <type_traits>
#include <vector>

#include "flat_hash_map.h"
#include "string_view.h"
#include "span.h"
#include "hlo_opcode.h"
#include "literal.h"
#include "status.h"
#include "types.h"
#include "xla_data.pb.h"
#include "status.h"

namespace xla {

class HloComputation;
class HloInstruction;

// A postorder depth-first HloInstruction visitor. When Handle* is called on an
// instruction, all its operands were already visited. User code can subclass
// this to iterate over an HloInstruction DAG. The Handle* routines have
// operands / data unpacked for ease of use in the visitor subclass.
//
// No instruction will ever be visited twice; however, the root instruction will
// be reported again when the traversal is done via a call to FinishVisit.
//
// A subclass must override at least
// (either HandleElementwiseUnary or all the Handle methods for unary ops) and
// (either HandleElementwiseBinary or all the Handle methods for binary ops)).
// The default Handle methods for (unary, binary) ops call
// (HandleElementwiseUnary, HandleElementwiseBinary).
// The default (HandleElementwiseUnary, HandleElementwiseBinary) return an
// "unimplemented" error status.
//
// Note: this may change to an iterator in the future for flexibility purposes.
//
// Users should not use this class directly, but use the type-aliases
// DfsHloVisitor/ConstDfsHloVisitor instead.
template <typename HloInstructionPtr>
class DfsHloVisitorBase {
  static_assert(
      std::is_same<HloInstruction*, HloInstructionPtr>::value ||
          std::is_same<const HloInstruction*, HloInstructionPtr>::value,
      "Template argument expected to be HloInstruction* or const "
      "HloInstruction*");

 public:
  DfsHloVisitorBase() = default;
  virtual ~DfsHloVisitorBase() = default;

  // These routines are self-descriptive, see class comment for usage
  // information.

  virtual Status HandleElementwiseUnary(HloInstructionPtr hlo);
  virtual Status HandleElementwiseBinary(HloInstructionPtr hlo);

  virtual Status HandleClamp(HloInstructionPtr hlo) = 0;
  virtual Status HandleSelect(HloInstructionPtr hlo) = 0;
  virtual Status HandleMaximum(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleMinimum(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleConcatenate(HloInstructionPtr hlo) = 0;
  virtual Status HandleConvert(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleBitcastConvert(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleStochasticConvert(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleCopy(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleComplex(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleMultiply(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleDot(HloInstructionPtr hlo) = 0;
  virtual Status HandlePower(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleSqrt(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleRsqrt(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleCbrt(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  /* go/keep-sorted start */
  virtual Status HandleAllGather(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllGatherDone(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllGatherStart(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllReduce(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllReduceDone(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllReduceStart(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllToAll(HloInstructionPtr hlo) = 0;
  virtual Status HandleCollectivePermute(HloInstructionPtr hlo) = 0;
  virtual Status HandleCollectivePermuteDone(HloInstructionPtr hlo) = 0;
  virtual Status HandleCollectivePermuteStart(HloInstructionPtr hlo) = 0;
  virtual Status HandleConvolution(HloInstructionPtr hlo) = 0;
  virtual Status HandleOptimizationBarrier(HloInstructionPtr hlo) = 0;
  virtual Status HandlePartitionId(HloInstructionPtr hlo) = 0;
  virtual Status HandleReduceScatter(HloInstructionPtr hlo) = 0;
  virtual Status HandleReplicaId(HloInstructionPtr hlo) = 0;
  /* go/keep-sorted end */

  /* go/keep-sorted start */
  virtual Status HandleCholesky(HloInstructionPtr hlo) = 0;
  virtual Status HandleFft(HloInstructionPtr fft) = 0;
  virtual Status HandleTopK(HloInstructionPtr hlo) = 0;
  virtual Status HandleTriangularSolve(HloInstructionPtr hlo) = 0;
  /* go/keep-sorted end */

  virtual Status HandleGetDimensionSize(HloInstructionPtr hlo) = 0;
  virtual Status HandleSetDimensionSize(HloInstructionPtr hlo) = 0;

  virtual Status HandleCompare(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleAdd(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleDivide(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleRemainder(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleSubtract(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleAbs(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleAtan2(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleRound(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleRoundNearestEven(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleLogistic(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleSign(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleNegate(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleExp(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleExpm1(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleFloor(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleCeil(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleLog(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleClz(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleLog1p(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleCos(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleSin(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleTan(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleTanh(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleReal(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleImag(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleIsFinite(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleAnd(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleNot(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleOr(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleXor(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandlePopulationCount(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleShiftLeft(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleShiftRightArithmetic(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleShiftRightLogical(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }

  virtual Status HandleReducePrecision(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }

  virtual Status HandleDomain(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }

  /* go/keep-sorted start */
  virtual Status HandleInfeed(HloInstructionPtr hlo) = 0;
  virtual Status HandleOutfeed(HloInstructionPtr hlo) = 0;
  /* go/keep-sorted end */

  /* go/keep-sorted start */
  virtual Status HandleBitcast(HloInstructionPtr hlo) = 0;
  virtual Status HandleBroadcast(HloInstructionPtr hlo) = 0;
  virtual Status HandleCall(HloInstructionPtr hlo) = 0;
  virtual Status HandleConditional(HloInstructionPtr hlo) = 0;
  virtual Status HandleConstant(HloInstructionPtr hlo) = 0;
  virtual Status HandleCustomCall(HloInstructionPtr hlo) = 0;
  virtual Status HandleDynamicReshape(HloInstructionPtr hlo) = 0;
  virtual Status HandleDynamicSlice(HloInstructionPtr hlo) = 0;
  virtual Status HandleDynamicUpdateSlice(HloInstructionPtr hlo) = 0;
  virtual Status HandleFusion(HloInstructionPtr hlo) = 0;
  virtual Status HandleGather(HloInstructionPtr hlo) = 0;
  virtual Status HandleGetTupleElement(HloInstructionPtr hlo) = 0;
  virtual Status HandleIota(HloInstructionPtr hlo) = 0;
  virtual Status HandleMap(HloInstructionPtr hlo) = 0;
  virtual Status HandleParameter(HloInstructionPtr hlo) = 0;
  virtual Status HandleReduce(HloInstructionPtr hlo) = 0;
  virtual Status HandleReduceWindow(HloInstructionPtr hlo) = 0;
  virtual Status HandleReshape(HloInstructionPtr hlo) = 0;
  virtual Status HandleReverse(HloInstructionPtr hlo) = 0;
  virtual Status HandleRng(HloInstructionPtr hlo) = 0;
  virtual Status HandleRngBitGenerator(HloInstructionPtr hlo) = 0;
  virtual Status HandleRngGetAndUpdateState(HloInstructionPtr hlo) = 0;
  virtual Status HandleScatter(HloInstructionPtr hlo) = 0;
  virtual Status HandleSelectAndScatter(HloInstructionPtr hlo) = 0;
  virtual Status HandleSlice(HloInstructionPtr hlo) = 0;
  virtual Status HandleSort(HloInstructionPtr hlo) = 0;
  virtual Status HandleTranspose(HloInstructionPtr hlo) = 0;
  virtual Status HandleTuple(HloInstructionPtr hlo) = 0;
  virtual Status HandleWhile(HloInstructionPtr hlo) = 0;
  /* go/keep-sorted end */

  virtual Status HandlePad(HloInstructionPtr hlo) = 0;

  virtual Status HandleAsyncStart(HloInstructionPtr hlo) = 0;
  virtual Status HandleAsyncUpdate(HloInstructionPtr hlo) = 0;
  virtual Status HandleAsyncDone(HloInstructionPtr hlo) = 0;

  virtual Status HandleCopyStart(HloInstructionPtr copy_start) = 0;
  virtual Status HandleCopyDone(HloInstructionPtr copy_done) = 0;

  virtual Status HandleSend(HloInstructionPtr send) = 0;
  virtual Status HandleSendDone(HloInstructionPtr send_done) = 0;

  virtual Status HandleRecv(HloInstructionPtr recv) = 0;
  virtual Status HandleRecvDone(HloInstructionPtr recv_done) = 0;

  virtual Status HandleBatchNormTraining(HloInstructionPtr hlo) = 0;

  virtual Status HandleBatchNormInference(HloInstructionPtr hlo) = 0;

  virtual Status HandleBatchNormGrad(HloInstructionPtr hlo) = 0;

  virtual Status HandleAddDependency(HloInstructionPtr add_dependency) = 0;
  virtual Status HandleAfterAll(HloInstructionPtr token) = 0;

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  virtual Status FinishVisit(HloInstructionPtr root) = 0;

  // 3 possible visitation states of HLO instructions. Each instruction's
  // state only flows one way: kNotVisited -> kVisiting -> kVisited.
  enum VisitState {
    kNotVisited = 0,
    kVisiting = 1,
    kVisited = 2,
  };

  VisitState GetVisitState(int id) {
    auto iter = visit_state_.find(id);
    if (iter == visit_state_.end()) {
      return VisitState::kNotVisited;
    }
    return iter->second;
  }
  VisitState GetVisitState(const HloInstruction& instruction);

  // Resize internal state if necessary to hold state for ids <= num.
  // This call is purely a performance hint and can be omitted without
  // affecting correctness.
  void ReserveVisitStates(int num) { visit_state_.reserve(num); }
  size_t VisitStateCapacity() const { return visit_state_.capacity(); }

  // Useful when we want to visit the same computation more than once with the
  // same visitor.
  void ResetVisitStates() {
    // Clear the map, but don't resize the capacity across uses -- Calculating
    // and reserving space could be expensive, and we always use the same
    // module->instruction_count() as the capacity.
    visit_state_.erase(visit_state_.begin(), visit_state_.end());
  }

  // Useful when we want to free up the memory used by the visit state without
  // destroying the actual visitor subclass.
  void DestroyVisitState() {
    visit_state_ = absl::flat_hash_map<int, VisitState>{};
  }

  void SetVisitState(int id, VisitState state) { visit_state_[id] = state; }

  // Sets the visitation state of the given instruction as kVisiting.
  //
  // Precondition: current state must be kNotVisited.
  void SetVisiting(const HloInstruction& instruction);

  // Sets the visitation state of the given instruction as kVisited.
  //
  // Precondition: current state must be either kNotVisited or kVisiting.
  void SetVisited(const HloInstruction& instruction);

  // Returns whether the state of the given instruction is kVisiting.
  bool IsVisiting(const HloInstruction& instruction) {
    return GetVisitState(instruction) == kVisiting;
  }

  // Returns whether the state of the given instruction is kVisited.
  bool DidVisit(const HloInstruction& instruction) {
    return GetVisitState(instruction) == kVisited;
  }

  // Returns whether the state of the given instruction is kNotVisited.
  bool NotVisited(const HloInstruction& instruction) {
    return GetVisitState(instruction) == kNotVisited;
  }

  // This method should be overridden by subclasses that wish to run some
  // operation on an op before its Handle* visitor method is called.
  //
  // For any HLO op, the order of calls is:
  //
  //   Preprocess(op);
  //   Handle/OpType/(op);
  //   Postprocess(op);
  //
  // Overriding methods should call DfsHloVisitor::Preprocess before doing their
  // own preprocessing.
  virtual Status Preprocess(HloInstructionPtr hlo);

  // This method should be overridden by subclasses that wish to run some
  // operation on an op after its Handle* visitor method is called. See
  // Preprocess for more details.
  //
  // Overriding methods should call DfsHloVisitor::Postprocess after doing their
  // own postprocessing.
  virtual Status Postprocess(HloInstructionPtr hlo);

 private:
  absl::flat_hash_map<int, VisitState> visit_state_;

  DfsHloVisitorBase(const DfsHloVisitorBase&) = delete;
  DfsHloVisitorBase& operator=(const DfsHloVisitorBase&) = delete;
};

// Explicit instantiations in dfs_hlo_visitor.cc.
extern template class DfsHloVisitorBase<HloInstruction*>;
extern template class DfsHloVisitorBase<const HloInstruction*>;

// Users should use one of these two type aliases, which are the only two valid
// instantiations of DfsHloVisitorBase.
using DfsHloVisitor = DfsHloVisitorBase<HloInstruction*>;
using ConstDfsHloVisitor = DfsHloVisitorBase<const HloInstruction*>;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_IR_DFS_HLO_VISITOR_H_
