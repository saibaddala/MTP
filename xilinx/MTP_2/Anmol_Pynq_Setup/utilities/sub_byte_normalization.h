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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SUB_BYTE_NORMALIZATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SUB_BYTE_NORMALIZATION_H_

#include "flat_hash_set.h"
#include "string_view.h"
#include "hlo_module.h"
#include "hlo_pass_interface.h"
#include "statusor.h"

namespace xla {

// A pass that unconditionally removes the sub-byte element_size_in_bits
// annotation for platforms that doesn't support nibble-packed types. After this
// pass, a sub-byte type is treated as int8 for space occupation and arithmetic
// operations. This pass is used in HloEvaluation and testing only.
class SubByteNormalization : public HloModulePass {
 public:
  SubByteNormalization() = default;

  ~SubByteNormalization() override = default;

  absl::string_view name() const override { return "int4-size-removal"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SUB_BYTE_NORMALIZATION_H_
