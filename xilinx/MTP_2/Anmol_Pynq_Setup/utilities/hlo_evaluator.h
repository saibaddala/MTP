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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EVALUATOR_HLO_EVALUATOR_H_
#define TENSORFLOW_COMPILER_XLA_HLO_EVALUATOR_HLO_EVALUATOR_H_

#define _USE_MATH_DEFINES

#include <functional>
#include <memory>
#include <optional>

#include "flat_hash_map.h"
#include "node_hash_map.h"
#include "span.h"
#include "array2d.h"
#include "dfs_hlo_visitor_with_default.h"
#include "hlo_computation.h"
#include "hlo_instruction.h"
#include "hlo_module.h"
#include "literal.h"
#include "literal_util.h"
#include "call_graph.h"
#include "dynamic_dimension_inference.h"
#include "shape_inference.h"
#include "tuple_points_to_analysis.h"
#include "shape_util.h"
#include "statusor.h"
#include "util.h"
#include "xla_data.pb.h"

namespace xla {

// Represents a parsed static while loop. We normalize the loop representation
// so that it starts from the induction_var_init_value and increments by
// step_size until it exceeds or goes below loop_bound.
struct ParsedStaticWhileLoop {
  // The number of iterations to be executed.
  int64_t trip_count = -1;
  // The tuple index of the induction variable in the while argument tuple.
  int64_t induction_var_index = -1;
  // The induction variable's initial value.
  int64_t induction_var_init_value = -1;
  // The induction variable is incremented by this number (could be negative)
  // in each iteration.
  int64_t step_size = -1;
  int64_t loop_bound = -1;
};

// Indicates whether a parsed while loop is static or dynamic. If the loop is
// static, it contains a value for StaticLoopInfo; otherwise the loop is
// dynamic. We consider a loop dynamic if its induction variable's initial
// value or the loop bound's value depends on the while's parent computation's
// parameter.
struct ParsedWhileLoop {
  std::optional<ParsedStaticWhileLoop> static_while_loop;
  bool is_dynamic() const { return !static_while_loop.has_value(); }
};
constexpr ParsedWhileLoop kParsedDynamicWhileLoop = ParsedWhileLoop();

// Tries to parse a while loop using a set of predefined patterns.
// Returns the parsing result.
std::optional<ParsedWhileLoop> PatternMatchParseWhileLoop(
    const HloInstruction* while_op);

// Responsible for evaluating HLO and obtain literal as the evaluation results.
//
// This class is not thread-safe.
class HloEvaluator : public ConstDfsHloVisitorWithDefault {
 public:
  // Only evaluate up to max_loop_iterations per while-loop execution if
  // specified.
  explicit HloEvaluator(int64_t max_loop_iterations = -1);

  // Called by the evaluator to create an embedded evaluator to execute a
  // sub-region of control flow. Subclasses should override this to return an
  // instance of the subclass instead.
  virtual std::unique_ptr<HloEvaluator> CreateEmbedded(
      int64_t max_loop_iterations) {
    return std::make_unique<HloEvaluator>(max_loop_iterations);
  }

  // Enables subclasses to be notified when a new computation is being
  // evaluated.
  virtual void OnEvaluateComputation(const HloComputation& computation) {}

  // Evaluates an HLO module and an array of pointers to literals.  Returns the
  // evaluated result as a literal if successful.
  //
  // Precondition: The indices of arg_literals correspond to the parameter
  // numbers of the HLO parameters in the computation. See comment below for an
  // example.
  //
  // (Dummy template arg is to reduce the overloading priority of one overload
  // so that Evaluate(module, {}) resolves unambiguously.)
  StatusOr<Literal> Evaluate(const HloModule& module,
                             absl::Span<const Literal* const> arg_literals) {
    return Evaluate(*module.entry_computation(), arg_literals);
  }
  template <typename Dummy = void>
  StatusOr<Literal> Evaluate(const HloModule& module,
                             absl::Span<const Literal> arg_literals) {
    return Evaluate(*module.entry_computation(), arg_literals);
  }

  // Evaluates an HLO computation and an array of pointers to literals.
  // Returns the evaluated result as a literal if successful.
  // Precondition: The indices of arg_literals correspond to the parameter
  // numbers of the HLO parameters in the computation. For e.g., consider the
  // following graph:
  //
  //                *
  //            /       \
  //            +     Parameter1
  //        /      \
  //       /        \
  //    Parameter0  Constant
  //
  // where Parameter0 has parameter_number 0 and Parameter1 has parameter_number
  // 1 in this computation. The input literals array will then have its first
  // literal map to Parameter0 and the second map to Parameter1.
  //
  // (Dummy template arg is to reduce the overloading priority of one overload
  // so that Evaluate(module, {}) resolves unambiguously.)
  StatusOr<Literal> Evaluate(const HloComputation& computation,
                             absl::Span<const Literal* const> arg_literals);
  template <typename Dummy = void>
  StatusOr<Literal> Evaluate(const HloComputation& computation,
                             absl::Span<const Literal> arg_literals) {
    std::vector<const Literal*> arg_literal_ptrs;
    for (const auto& l : arg_literals) {
      arg_literal_ptrs.push_back(&l);
    }
    return Evaluate(computation, arg_literal_ptrs);
  }

  // Gets the value of running a single HLO instruction.
  //
  // This function may recursively evaluate the dependency of this instruction
  // within its parent computation until it encounters something that cannot be
  // evaluated, such as an Infeed or a Parameter instruction.
  // It makes best effort to partially evaluate a dependency if possible.
  StatusOr<Literal> Evaluate(
      const HloInstruction* instruction,
      bool recursively_evaluate_nonconstant_operands = false);

  // Same as Evaluate, except returning false on error and accepts an output
  // pointer.
  bool TryEvaluate(const HloInstruction* instruction, Literal* result,
                   bool recursively_evaluate_nonconstant_operands = false);

  // Evaluates a single HLO instruction, substituting the given literals for
  // some of the instruction's operands.
  //
  // For example, given instruction = op(A, B, C) and the map
  // {A = x, C = y}, this evaluates op(x, B, y).
  StatusOr<Literal> EvaluateWithSubstitutions(
      const HloInstruction* instruction,
      const absl::flat_hash_map<const HloInstruction*, const Literal*>&
          substitutions);

  StatusOr<Literal> EvaluateElementwiseBinaryOp(HloOpcode opcode,
                                                const Literal& lhs,
                                                const Literal& rhs);

  StatusOr<Literal> EvaluateElementwiseUnaryOp(HloOpcode opcode,
                                               const Literal& operand);

  StatusOr<Literal> EvaluateElementwiseTernaryOp(HloOpcode opcode,
                                                 const Literal& lhs,
                                                 const Literal& rhs,
                                                 const Literal& ehs);

  StatusOr<Literal> EvaluateElementwiseCompareOp(ComparisonDirection direction,
                                                 const Literal& lhs,
                                                 const Literal& rhs);

  StatusOr<Literal> EvaluateDotOp(const DotDimensionNumbers& dim_numbers,
                                  const PrecisionConfig& precision_config,
                                  const Literal& lhs, const Literal& rhs);

  void set_dynamic_dimension_inference(
      DynamicDimensionInference* dynamic_dimension_inference) {
    dynamic_dimension_inference_ = dynamic_dimension_inference;
  }

  DynamicDimensionInference* dynamic_dimension_inference() {
    return dynamic_dimension_inference_;
  }

  // Enable the fast path for certain operations like dot or convolution.
  void set_use_fast_path(bool value) { use_fast_path_ = value; }

  // Handles evaluation of a custom-call op.
  // Operand literals are provided in |operands| and implementations must
  // populate |output| before returning.
  using CustomCallHandler = std::function<StatusOr<Literal>(
      const HloInstruction* custom_call, absl::Span<const Literal*> operands)>;

  // Sets a handler that is called during evaluation for custom-call ops.
  // If no handler is defined the default error behavior will occur. The handler
  // will be provided evaluated literals for all operands and is expected to
  // return an output literal of the appropriate shape.
  void set_custom_call_handler(CustomCallHandler handler) {
    custom_call_handler_ = std::move(handler);
  }

  // Returns the result of a matrix multiply `lhs x rhs`.
  static std::unique_ptr<Array2D<Eigen::half>> MatmulArray2D(
      const Array2D<Eigen::half>& lhs, const Array2D<Eigen::half>& rhs);
  static std::unique_ptr<Array2D<float>> MatmulArray2D(
      const Array2D<float>& lhs, const Array2D<float>& rhs);
  static std::unique_ptr<Array2D<double>> MatmulArray2D(
      const Array2D<double>& lhs, const Array2D<double>& rhs);
  static std::unique_ptr<Array2D<std::complex<float>>> MatmulArray2D(
      const Array2D<std::complex<float>>& lhs,
      const Array2D<std::complex<float>>& rhs);
  static std::unique_ptr<Array2D<std::complex<double>>> MatmulArray2D(
      const Array2D<std::complex<double>>& lhs,
      const Array2D<std::complex<double>>& rhs);
  static std::unique_ptr<Array2D<int32_t>> MatmulArray2D(
      const Array2D<int32_t>& lhs, const Array2D<int32_t>& rhs);

 protected:
  // Evaluates the given instruction, and stores the evaluation result in the
  // evaluated_ map.
  // When a non-empty shape_index is given, the instruction may be partially
  // evaluated at the given shape_index and the rest of the result could be
  // marked as undetermined unless it has been previously evaluated using
  // EvaluateInternal. Such partial evaluation reduces the computation and
  // memory overhead in cases where we need only one tuple element by avoiding
  // the evaluation of a full tuple.
  Status EvaluateInternal(
      const HloInstruction* instruction, const ShapeIndex& shape_index = {},
      bool recursively_evaluate_nonconstant_operands = false);

  Status EvaluateParameterFromCallerArgument(const HloInstruction* parameter,
                                             const ShapeIndex& shape_index);

  // Helper method to extract a list of int64_t from evaluated instruction for
  // start_indices for DynamicSlice and DynamicUpdateSlice.
  std::vector<int64_t> GetS64Indices(
      absl::Span<HloInstruction* const> start_indices);

  // Creates a vector of multipliers which can be used to create a linear index
  // into shape.
  //
  // Given the multidimensional index {i1, ..., iN} and
  // M = MakeDimMultipliers(shape), the corresponding linear index LI is simply
  //
  //   LI = i1 * M[1] + i2 * M[2] + ... + iN * M[N].
  //
  // This lets you calculate LI given the multidimensional indices in any order.
  static DimensionVector MakeDimMultipliers(const Shape& shape);

  // Make HloEvaluatorTypedVisitor a friend because it is logically part of this
  // class.
  //
  // A straightforward implementation would be to make it a nested class
  // declared and defined in hlo_evaluator.cc.  Instead HloEvaluatorTypedVisitor
  // lives as a separate class with its own header because its template gets
  // instantiated many times and we want to use extern templates to shard out
  // the compilation of those instantiations across multiple cc files.
  template <typename ReturnT, typename ElementwiseT>
  friend class HloEvaluatorTypedVisitor;

  // Wraps around instruction handling to infer types before dispatching to
  // the corresponding typed Visitor.
  Status DefaultAction(const HloInstruction* hlo) override {
    return hlo->Visit(typed_visitors_[hlo->shape().element_type()].get());
  }

  Status Preprocess(const HloInstruction* hlo) override;
  Status Postprocess(const HloInstruction* hlo) override;

  // Operations that are type-agnostic or always return a specific type, such as
  // HandleIsFinite where boolean is always returned.
  //
  Status HandleBitcast(const HloInstruction* bitcast) override;
  Status HandleBitcastConvert(const HloInstruction* convert) override;
  Status HandleGetDimensionSize(
      const HloInstruction* get_dimension_size) override;
  Status HandleSetDimensionSize(
      const HloInstruction* set_dimension_size) override;
  Status HandleParameter(const HloInstruction* parameter) override;
  Status HandleInfeed(const HloInstruction* infeed) override;
  Status HandleConstant(const HloInstruction* constant) override;
  Status HandleConcatenate(const HloInstruction* concatenate) override;
  Status HandleReshape(const HloInstruction* reshape) override;
  Status HandleTranspose(const HloInstruction* transpose) override;
  Status HandleIsFinite(const HloInstruction* is_finite) override;
  Status HandleCompare(const HloInstruction* compare) override;
  Status HandleTuple(const HloInstruction* tuple) override;
  Status HandleFft(const HloInstruction* fft) override;
  Status HandleGather(const HloInstruction* gather) override;
  Status HandleScatter(const HloInstruction* hlo) override;
  Status HandleGetTupleElement(
      const HloInstruction* get_tuple_element) override;
  Status HandleAsyncStart(const HloInstruction* async_start) override;
  Status HandleAsyncUpdate(const HloInstruction* async_update) override;
  Status HandleAsyncDone(const HloInstruction* async_done) override;
  Status HandleCopy(const HloInstruction* copy) override;
  Status HandleCopyStart(const HloInstruction* copy_start) override;
  Status HandleCopyDone(const HloInstruction* copy_done) override;
  Status HandleConditional(const HloInstruction* conditional) override;
  Status HandleConvert(const HloInstruction* convert) override;
  Status HandleCall(const HloInstruction* call) override;
  Status HandleDynamicSlice(const HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(const HloInstruction* dus) override;
  Status HandleFusion(const HloInstruction* fusion) override;
  Status HandleWhile(const HloInstruction* while_hlo) override;
  Status HandleSelect(const HloInstruction* select) override;
  Status HandleBroadcast(const HloInstruction* broadcast) override;
  Status HandleAfterAll(const HloInstruction* after_all) override;
  Status HandleAddDependency(const HloInstruction* add_dependency) override;
  Status HandleReverse(const HloInstruction* reverse) override;
  Status HandleSelectAndScatter(
      const HloInstruction* select_and_scatter) override;
  Status HandleSlice(const HloInstruction* slice) override;
  Status HandleSort(const HloInstruction* sort) override;
  Status HandleStochasticConvert(
      const HloInstruction* stochastic_convert) override;
  Status HandleReal(const HloInstruction* real) override;
  Status HandleImag(const HloInstruction* imag) override;
  Status HandleComplex(const HloInstruction* complex) override;
  Status HandleReduce(const HloInstruction* hlo) override;
  Status HandleReduceWindow(const HloInstruction* hlo) override;
  Status HandleMap(const HloInstruction* map) override;
  Status HandleCustomCall(const HloInstruction* custom_call) override;

  // Unsupported HLOs, note some of them (such as BatchNorm*) are typically
  // expanded in a semantic-preserving way into other HLOs by adding expansion
  // HLO pass to the HLO optimization pass during compilation, which can then be
  // handled by the evaluator.
  Status HandleBatchNormGrad(const HloInstruction* batch_norm_grad) override {
    return Unimplemented("BatchNormGrad HLO is unsupported by the evaluator.");
  }
  Status HandleBatchNormInference(
      const HloInstruction* batch_norm_inference) override {
    return Unimplemented(
        "BatchNormInference HLO is unsupported by the evaluator.");
  }
  Status HandleBatchNormTraining(
      const HloInstruction* batch_norm_training) override {
    return Unimplemented(
        "BatchNormTraining HLO is unsupported by the evaluator.");
  }
  Status HandleOutfeed(const HloInstruction* outfeed) override {
    return Unimplemented("Outfeed HLO is unsupported by the evaluator.");
  }

  // Returns the already-evaluated literal result for the instruction.
  //
  // A Constant instruction is considered evaluated and its literal will be
  // returned directly without looking up the cache.
  //
  // Similarly, a Parameter instruction is considered evaluated and its literal
  // is looked up in arg_literals.
  //
  // Crash with log if the given instruction has not been evaluated previously.
  const Literal& GetEvaluatedLiteralFor(const HloInstruction* hlo) {
    if (hlo->IsConstant()) {
      return hlo->literal();
    }
    if (hlo->opcode() == HloOpcode::kParameter && !arg_literals_.empty()) {
      return *arg_literals_.at(hlo->parameter_number());
    }

    auto it = evaluated_.find(hlo);
    CHECK(it != evaluated_.end())
        << "could not find evaluated value for: " << hlo->ToString();
    return it->second;
  }

  // Returns true if the given hlo has been evaluated and cached.
  bool IsAlreadyEvaluated(const HloInstruction* hlo,
                          const ShapeIndex& shape_index = {}) {
    if (hlo->IsConstant()) {
      return true;
    }
    if (hlo->opcode() == HloOpcode::kParameter && !arg_literals_.empty()) {
      return true;
    }
    auto it = evaluated_.find(hlo);
    if (it == evaluated_.end()) {
      return false;
    }
    // We may evaluate some elements of a tuple-shaped instruction and mark
    // the other elements as undetermined. This way we avoid the computation
    // and memory overhead of evaluating a large tuple when only some elements
    // are needed. By marking the other elements undetermined, we allow the
    // evaluator to update the cached tuple literal when more elements are
    // evaluated.
    return it->second.IsDetermined(shape_index);
  }

  // Tracks the HLO instruction and its evaluated literal result.
  //
  // Parameters and constants aren't stored here, see implementation of
  // GetEvaluatedLiteralFor.
  //
  // TODO(b/35950897): have better memory management here to free instructions
  // that are no longer a parent for any other subsequent instruction in
  // post-ordering.
  //
  // Must be cleared for each evaluation.
  //
  // Storing Literal in place requires the container to have pointer stability
  // so we cannot use flat_hash_map any more.
  absl::node_hash_map<const HloInstruction*, Literal> evaluated_;
  // Set by EvaluateInternal and opportunitiscally used by the HandleXXX
  // functions. When non-empty, the HandleXXX function may evaluate the
  // instruction at only the given shape index.
  ShapeIndex visitor_shape_index_;
  bool enable_partial_evaluation_ = false;

  std::unique_ptr<CallGraph> call_graph_cache_;
  std::unique_ptr<TuplePointsToAnalysis> tuple_points_to_analysis_cache_;

  // Use fast path that uses eigen in the evaluator.
  bool use_fast_path_ = false;

 private:
  template <typename ReturnT, typename NativeT>
  static StatusOr<Literal> ElementWiseUnaryOpImpl(
      const HloInstruction* instruction,
      const std::function<ReturnT(NativeT)>& unary_op,
      const Literal& operand_literal) {
    const Shape& shape = instruction->shape();
    const auto* operand = instruction->operand(0);
    TF_RET_CHECK(ShapeUtil::SameDimensions(shape, operand->shape()));

    Literal result(shape);
    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(
        [&](absl::Span<const int64_t> multi_index, int) {
          return unary_op(operand_literal.Get<NativeT>(multi_index));
        }));
    return std::move(result);
  }

  // Map from a primitive type to its associated (templated) DfsHloVisitor.
  std::unique_ptr<ConstDfsHloVisitor> typed_visitors_[PrimitiveType_ARRAYSIZE];

  // Caches pointers to input literals, assuming they are in post-order.
  // Literals are not owned by this class, and they must outlive the lifetime of
  // each invocation to the Evaluate* method.
  // Must be cleared for each evaluation.
  std::vector<const Literal*> arg_literals_;

  // Max loop iterations to execute with no maximum if negative.
  int64_t max_loop_iterations_ = 0;

  // Module-level seed handle.
  uint64_t seed_ = 0;
  // RNG engine.
  std::minstd_rand0 engine_;

  // DynamicDimensionInference is used to evaluate GetDimensionSize, which
  // returns the dynamic dimension size of its operand.
  DynamicDimensionInference* dynamic_dimension_inference_ = nullptr;

  // Optional handler for custom_call ops.
  CustomCallHandler custom_call_handler_;

  HloEvaluator(const HloEvaluator&) = delete;
  HloEvaluator& operator=(const HloEvaluator&) = delete;
};

std::unique_ptr<Array2D<float>> MatmulArray2D(const Array2D<float>& lhs,
                                              const Array2D<float>& rhs);
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_EVALUATOR_HLO_EVALUATOR_H_
