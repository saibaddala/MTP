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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_RUNNER_PJRT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_RUNNER_PJRT_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "pjrt_client.h"
#include "hlo_runner_interface.h"

namespace xla {

// A base class for running an HloModule using the PjRt API. This class
// abstracts execution for a given HloModule using PjRt interfaces.
// HloModule can be explicitly built, or loaded from a serialization file (e.g.,
// hlo proto file), or parsed from a hlo textual IR string.
class HloRunnerPjRt : public HloRunnerInterface {
 public:
  explicit HloRunnerPjRt(
      std::unique_ptr<PjRtClient> pjrt_client,
      DeviceShapeRepresentationFn device_shape_representation_fn);

  ~HloRunnerPjRt() override;

  // Transfers data between the host and device.
  StatusOr<std::unique_ptr<PjRtBuffer>> TransferLiteralToDevice(
      const Literal& literal);
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> TransferLiteralsToDevice(
      absl::Span<const Literal* const> literals);
  StatusOr<Literal> TransferLiteralFromDevice(PjRtBuffer& buffer);

  // Executes the given module with given literals as input and returns the
  // result as a Literal.
  StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                            absl::Span<const Literal* const> arguments,
                            bool run_hlo_passes,
                            ExecutionProfile* profile) override;

  // As Execute(), but accepts and returns device buffers instead of host
  // buffers.
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteWithDeviceBuffers(
      PjRtLoadedExecutable* executable,
      const std::vector<std::unique_ptr<PjRtBuffer>>& arguments);

  // Creates an executable object for an HloModule.
  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CreateExecutable(
      HloModule* module, CompileOptions compile_options);

  // Creates an executable object given an HLO module. If run_hlo_passes is
  // true, the HLO passes will be run as part of compilation.
  StatusOr<std::unique_ptr<Executable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) override;

  StatusOr<Literal> ExecuteWithExecutable(
      Executable* executable, absl::Span<const Literal* const> arguments,
      ExecutionProfile* profile) override;

  StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options) override;

  // Same as above, but with specified device assignment.
  StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override;

  StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::function<Executable*(int64_t)> executable_provider,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override;

  StatusOr<std::vector<Literal>> ExecuteReplicated(
      Executable* executable,
      const HloRunnerInterface::ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment, ExecutionProfile* profile = nullptr);

  absl::string_view Name() const override;

 private:
  std::unique_ptr<PjRtClient> pjrt_client_;
  DeviceShapeRepresentationFn device_shape_representation_fn_;

  std::vector<PjRtBuffer*> BufferVecToPointerVec(
      const std::vector<std::unique_ptr<PjRtBuffer>>& buffer);

  std::vector<std::vector<PjRtBuffer*>> BufferMatToPointerMat(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>& buffer);

  StatusOr<CompileOptions> GenerateDefaultCompileOptions(HloModule* module,
                                                         bool run_hlo_passes);

  StatusOr<std::vector<Literal>> ExecuteReplicatedImpl(
      std::function<StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>(
          absl::Span<const std::vector<PjRtBuffer*>>&)>
          execution_helper,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_RUNNER_PJRT_H_
