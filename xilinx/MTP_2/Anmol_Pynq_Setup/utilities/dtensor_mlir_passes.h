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

#ifndef TENSORFLOW_DTENSOR_MLIR_DTENSOR_MLIR_PASSES_H_
#define TENSORFLOW_DTENSOR_MLIR_DTENSOR_MLIR_PASSES_H_

#include <memory>

#include "SparseTensor.h"  // from @llvm-project
#include "BuiltinOps.h"  // from @llvm-project
#include "Pass.h"  // from @llvm-project
#include "PassManager.h"  // from @llvm-project
#include "passes.h"
#include "status.h"

namespace tensorflow {
namespace dtensor {

bool MaybeEnableLogging(mlir::PassManager* pm);

// Adds MLIR passes to `pm`.
void CreateDTensorMLIRPass(const mlir::TF::StandardPipelineOptions& options,
                           mlir::OpPassManager* pm);

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_DTENSOR_MLIR_PASSES_H_
