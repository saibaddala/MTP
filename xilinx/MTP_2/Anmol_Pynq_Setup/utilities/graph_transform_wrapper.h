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

#ifndef TENSORFLOW_CORE_TRANSFORMS_GRAPH_TRANSFORM_WRAPPER_H_
#define TENSORFLOW_CORE_TRANSFORMS_GRAPH_TRANSFORM_WRAPPER_H_

#include <initializer_list>
#include <memory>

#include "STLFunctionalExtras.h"
#include "Pass.h"  // from @llvm-project
#include "graph_debug_info.pb.h"
#include "graph.h"
#include "status.h"

namespace mlir {
namespace tfg {

// Runs a sequence of passes over Graph* and attached function library. The
// Graph* is converted to TFG, provided passes executed and the passed in Graph*
// replaced. If the pass fails, then graph is not modified.
//
// This is meant for simple interop where there is a Graph* currently. Passes
// created here are constrained to run on Module ops.
tensorflow::Status RunTransformOnGraph(
    tensorflow::Graph* graph,
    const std::initializer_list<
        llvm::function_ref<std::unique_ptr<mlir::Pass>()>>& passes,
    const tensorflow::GraphDebugInfo& debug_info = {});

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_TRANSFORMS_GRAPH_TRANSFORM_WRAPPER_H_
