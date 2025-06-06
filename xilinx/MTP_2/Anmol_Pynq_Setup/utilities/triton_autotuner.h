/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRITON_AUTOTUNER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRITON_AUTOTUNER_H_

#include <vector>

#include "flat_hash_set.h"
#include "string_view.h"
#include "autotuning.pb.h"
#include "hlo_computation.h"
#include "hlo_module.h"
#include "autotuner_util.h"
#include "hlo_pass_interface.h"
#include "threadpool.h"

namespace xla {
namespace gpu {

// Find best tiling configuration for each triton fusion outlined.
class TritonAutotuner : public HloModulePass {
 public:
  explicit TritonAutotuner(const AutotuneConfig& config,
                           tsl::thread::ThreadPool* thread_pool)
      : config_(config), thread_pool_(thread_pool) {}

  absl::string_view name() const override { return "triton-autotuner"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  AutotuneConfig config_;
  tsl::thread::ThreadPool* thread_pool_;
};

// TODO(b/266210099): have a way to generate/load these dynamically.
// Returns a list of possible tilings for a GEMM performed in Triton.
std::vector<AutotuneResult::TritonGemmKey> GetPossibleMatmulAutotuneConfigs(
    const HloInstruction& instr, se::CudaComputeCapability compute_capability,
    bool exhaustive_tiling_search = false);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRITON_AUTOTUNER_H_
