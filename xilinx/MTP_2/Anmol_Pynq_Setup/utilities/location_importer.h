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

#ifndef TENSORFLOW_COMPILER_XLA_TRANSLATE_HLO_TO_MHLO_LOCATION_IMPORTER_H_
#define TENSORFLOW_COMPILER_XLA_TRANSLATE_HLO_TO_MHLO_LOCATION_IMPORTER_H_

#include "Location.h"  // from @llvm-project
#include "hlo_instruction.h"

namespace mlir {
namespace mhlo {

// Returns an MLIR Location generated from HLO Instruction. Uses instruction
// metadata if present or instruction name.
mlir::Location GenerateInstructionLocation(
    const xla::HloInstruction* instruction, mlir::MLIRContext* context);

}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_TRANSLATE_HLO_TO_MHLO_LOCATION_IMPORTER_H_
