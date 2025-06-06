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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V0_COMPILE_TF_GRAPH_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V0_COMPILE_TF_GRAPH_H_

#include <variant>
#include <vector>

#include "status.h"
#include "variant.h"
#include "compile_only_client.h"
#include "compile_options.pb.h"
#include "tensor_shape.h"
#include "tpu_compile.pb.h"
#include "tpu_compile_op_support.h"

namespace tensorflow {
namespace tf2xla {
namespace v0 {

// Compiles the given Tensorflow graph into xla::HLO. The result is in
// compilation_result. If the input computation is in MLIR, it will be
// converted to a Tensorflow graph. Otherwise, the graph compiler will be run.
tsl::Status CompileTensorflowGraphToHlo(
    const std::variant<tpu::MlirToHloArgs, tpu::FunctionToHloArgs>& computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_funcs,
    const std::vector<tensorflow::TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    xla::CompileOnlyClient* client,
    XlaCompiler::CompilationResult* compilation_result);

}  // namespace v0
}  // namespace tf2xla
};  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V0_COMPILE_TF_GRAPH_H_
