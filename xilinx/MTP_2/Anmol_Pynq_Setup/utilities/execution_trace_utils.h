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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_TOOLS_MLIR_REPLAY_PUBLIC_EXECUTION_TRACE_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_TOOLS_MLIR_REPLAY_PUBLIC_EXECUTION_TRACE_UTILS_H_

#include "Attributes.h"  // from @llvm-project
#include "Operation.h"  // from @llvm-project
#include "Region.h"  // from @llvm-project
#include "LLVM.h"  // from @llvm-project
#include "literal.h"
#include "execution_trace.pb.h"
#include "interpreter.h"
#include "interpreter_value.h"
#include "xla_data.pb.h"
#include "statusor.h"

namespace mlir {
namespace interpreter {

// Interpreter listener that builds a trace of all executed ops and regions.
class ExecutionTraceListener : public InterpreterListener {
 public:
  explicit ExecutionTraceListener(ExecutionTrace* trace) : trace_(trace) {}

  void beforeOp(ArrayRef<InterpreterValue> args, Operation* op) override;
  void afterOp(ArrayRef<InterpreterValue> results) override;
  void enterRegion(ArrayRef<InterpreterValue> bbargs, Region& region) override;
  void leaveRegion(ArrayRef<InterpreterValue> yielded) override;

 private:
  ExecutionTrace* trace_;
  SmallVector<RegionTrace*> regions_;
};

// Returns an attribute with the given contents and type.
llvm::SmallVector<mlir::Attribute> ValueToAttribute(
    const InterpreterValue& value, mlir::Type type);

// Deserializes the given literal.
tsl::StatusOr<InterpreterValue> LiteralToValue(
    const xla::LiteralProto& literal);
// Deserializes the given literal and then casts it to the given type.
tsl::StatusOr<InterpreterValue> LiteralToValue(const xla::LiteralProto& literal,
                                               mlir::Type type);

// Deserializes the given literal.
tsl::StatusOr<InterpreterValue> LiteralToValue(const xla::Literal& literal);

// Serializes the given interpreter value.
TracedValue ValueToTracedValue(const InterpreterValue& value);

// Deserializes the given traced value.
tsl::StatusOr<InterpreterValue> TracedValueToValue(
    const TracedValue& traced_value);

// Returns all executions of the given op in the given trace.
llvm::SmallVector<const InstructionTrace*> FindOpExecutionsInTrace(
    const ExecutionTrace& trace, mlir::Operation* op);

}  // namespace interpreter
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_TOOLS_MLIR_REPLAY_PUBLIC_EXECUTION_TRACE_UTILS_H_
