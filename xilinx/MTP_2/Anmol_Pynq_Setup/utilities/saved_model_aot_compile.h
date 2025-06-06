/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_AOT_COMPILE_H_
#define TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_AOT_COMPILE_H_

#include <memory>
#include <string>
#include <unordered_set>

#include "compiler.h"
#include "graph_execution_options.h"

namespace tensorflow::tfrt_stub {
struct AotOptions {
  AotOptions();
  std::unordered_set<std::string> tags = {};

  std::shared_ptr<GraphExecutionOptions> graph_execution_options;
};

// AOT Compiles saved_model in input_model_dir, writing output
// saved_model and aot packages to output_model_dir, or
// "{input_model_dir}/aot_packages" if output dir provided. Warmup requests
// should be present in input_model_dir
Status AotCompileSavedModel(absl::string_view input_model_dir,
                            AotOptions aot_options = {},
                            absl::string_view output_model_dir = "");

}  // namespace tensorflow::tfrt_stub

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_AOT_COMPILE_H_
