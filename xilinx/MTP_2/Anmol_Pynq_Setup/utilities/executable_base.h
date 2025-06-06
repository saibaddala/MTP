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

#ifndef TENSORFLOW_COMPILER_XLA_BACKENDS_INTERPRETER_EXECUTABLE_BASE_H_
#define TENSORFLOW_COMPILER_XLA_BACKENDS_INTERPRETER_EXECUTABLE_BASE_H_

#include <memory>
#include <optional>

#include "hlo_module.h"
#include "literal.h"
#include "dynamic_dimension_inference.h"
#include "executable.h"
#include "hlo_execution_profile.h"
#include "service_executable_run_options.h"
#include "shape.h"
#include "statusor.h"
#include "xla.pb.h"
namespace xla {
namespace interpreter {

// Responsible for running a HLO graph through the HloEvaluator and output
// buffer allocation. Refer to interpreter/README.md for more.
class InterpreterExecutableBase : public Executable {
 public:
  explicit InterpreterExecutableBase(std::unique_ptr<HloModule> hlo_module);

  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

 protected:
  virtual StatusOr<Literal> Evaluate(
      const ServiceExecutableRunOptions* run_options,
      const HloComputation& computation,
      absl::Span<const Literal> arg_literals) = 0;

 private:
  StatusOr<ExecutionOutput> AllocateOutputMemoryWithInputReuse(
      const Shape& shape, const HloInputOutputAliasConfig& alias_config,
      se::DeviceMemoryAllocator* allocator,
      std::vector<ExecutionInput>* arguments, stream_executor::Stream* stream);

  InterpreterExecutableBase(const InterpreterExecutableBase&) = delete;
  InterpreterExecutableBase& operator=(const InterpreterExecutableBase&) =
      delete;
};

}  // namespace interpreter
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_BACKENDS_INTERPRETER_EXECUTABLE_BASE_H_
