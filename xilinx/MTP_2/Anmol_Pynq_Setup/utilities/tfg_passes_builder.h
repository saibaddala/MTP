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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_TFG_PASSES_BUILDER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_TFG_PASSES_BUILDER_H_

#include "PassManager.h"  // from @llvm-project
#include "rewriter_config.pb.h"

namespace mlir {
namespace tfg {

// Constructs the default graph/function-level TFG pass pipeline.
void DefaultGrapplerPipeline(PassManager& manager);

// Constructs the default module-level TFG pass pipeline.
void DefaultModuleGrapplerPipeline(PassManager& manager,
                                   const tensorflow::RewriterConfig& config);

// Constructs the Remapper pass pipeline.
void RemapperPassBuilder(PassManager& manager);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_TFG_PASSES_BUILDER_H_
