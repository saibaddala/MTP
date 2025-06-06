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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_GRAPHDEF_IMPORT_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_GRAPHDEF_IMPORT_H_

#include "BuiltinOps.h"  // from @llvm-project
#include "MLIRContext.h"  // from @llvm-project
#include "OwningOpRef.h"  // from @llvm-project
#include "function.h"
#include "graph.pb.h"
#include "graph_debug_info.pb.h"
#include "graph.h"
#include "statusor.h"

namespace mlir {
namespace tfg {

// Convert a GraphDef directly to TFG.
tensorflow::StatusOr<OwningOpRef<ModuleOp>> ImportGraphDef(
    MLIRContext *context, const tensorflow::GraphDebugInfo &debug_info,
    const tensorflow::GraphDef &graph_def);

// Converts a graph and function library to a TFG module.
tensorflow::StatusOr<OwningOpRef<ModuleOp>> ImportGraphAndFunctionsToMlir(
    MLIRContext *context, const tensorflow::GraphDebugInfo &debug_info,
    const tensorflow::Graph &graph,
    const tensorflow::FunctionLibraryDefinition &flib_def);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_GRAPHDEF_IMPORT_H_
