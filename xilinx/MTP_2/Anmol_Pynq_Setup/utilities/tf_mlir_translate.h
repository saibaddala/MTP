/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_H_

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "macros.h"
#include "string_view.h"
#include "span.h"
#include "FuncOps.h"  // from @llvm-project
#include "BuiltinOps.h"  // from @llvm-project
#include "MLIRContext.h"  // from @llvm-project
#include "loader.h"
#include "mlir_import_options.h"

namespace tensorflow {

using tsl::Status;
using tsl::StatusOr;

struct GraphdefToMlirOptions {
  std::string debug_info_file;
  std::string xla_compile_device_type;
  bool prune_unused_nodes;
  bool convert_legacy_fed_inputs;
  bool graph_as_function;
  bool upgrade_legacy;
  bool enable_shape_inference;
  bool unconditionally_use_set_output_shapes;
  bool enable_soft_placement;
  bool set_original_tf_func_name = false;
};

// TODO(antiagainst): Directly manipulating files in library functions is not
// a good idea. We should pass in a string/stream here.

// Converts a TensorFlow GraphDef contained in `input` param into a MLIR module.
// Creates MLIR entities into the given MLIR `context`.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GraphdefToMlirTranslateFunction(
    llvm::StringRef input, const std::vector<std::string>& input_arrays,
    const std::vector<std::string>& input_dtypes,
    const std::vector<std::optional<std::vector<int>>>& input_shapes,
    const std::vector<std::string>& output_arrays,
    const std::vector<std::string>& control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context);

ABSL_DEPRECATED(
    "Please use the other overload of this function which accepts structured "
    "inputs instead of strings")
// Converts a TensorFlow GraphDef contained in `input` param into a MLIR module.
// Creates MLIR entities into the given MLIR `context`.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GraphdefToMlirTranslateFunction(
    llvm::StringRef input, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, absl::string_view control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context);

// Similar as the above function, but replaces all constant tensors
// with randomly generated splat values.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GraphdefToSplattedMlirTranslateFunction(
    llvm::StringRef input, const std::vector<std::string>& input_arrays,
    const std::vector<std::string>& input_dtypes,
    const std::vector<std::vector<int>>& input_shapes,
    const std::vector<std::string>& output_arrays,
    const std::vector<std::string>& control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context);

ABSL_DEPRECATED(
    "Please use the other overload of this function which accepts structured "
    "inputs instead of strings")
// Similar as the above function, but replaces all constant tensors
// with randomly generated splat values.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GraphdefToSplattedMlirTranslateFunction(
    llvm::StringRef input, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, absl::string_view control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context);

// Converts a TensorFlow SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> SavedModelObjectGraphToMlirImport(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    bool unconditionally_use_set_output_shapes = false);

// Converts a TensorFlow V1 SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`.
// 'saved_model_bundle' if not null, will be initialized with the model bundle.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> SavedModelSignatureDefsToMlirImport(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    MLIRImportOptions options,
    std::unique_ptr<tensorflow::SavedModelBundle>* saved_model_bundle =
        nullptr);

// Converts a TensorFlow V1 SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`. This does not create session internally so it is faster
// and does not perform any graph transformation.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefsToMlirImportLite(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    MLIRImportOptions options);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_H_
