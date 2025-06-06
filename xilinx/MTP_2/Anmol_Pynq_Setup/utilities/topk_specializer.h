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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TOPK_SPECIALIZER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TOPK_SPECIALIZER_H_

#include "flat_hash_set.h"
#include "string_view.h"
#include "executable_run_options.h"
#include "hlo_module.h"
#include "hlo_pass_interface.h"
#include "statusor.h"
#include "statusor.h"

namespace xla::gpu {

// This pass transforms eligible TopK CustomCall into a call to be executed by
// runtime/topk.cc.
class TopkSpecializer : public HloModulePass {
 public:
  absl::string_view name() const override { return "topk-specializer"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TOPK_SPECIALIZER_H_
