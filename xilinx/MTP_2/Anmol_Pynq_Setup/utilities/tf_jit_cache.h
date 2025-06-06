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

#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_JIT_CACHE_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_JIT_CACHE_H_

#include <functional>
#include <string>

#include "flat_hash_map.h"
#include "ExecutionEngine.h"  // from @llvm-project
#include "resource_op_kernel.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

class JITCache : public tensorflow::ResourceBase {
 public:
  static constexpr const char* kDefaultResourceName = "mlir-jit-cache";
  static tensorflow::Status Create(JITCache** dst);

  std::string DebugString() const override;
  ExecutionEngine* LookupOrCompile(
      const std::string code,
      std::function<llvm::Expected<std::unique_ptr<ExecutionEngine>>()>
          compile_callback);
  size_t Size();

 private:
  tensorflow::mutex mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<ExecutionEngine>>
      execution_engine_by_key_ TF_GUARDED_BY(mu_);
};

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_JIT_CACHE_H_
