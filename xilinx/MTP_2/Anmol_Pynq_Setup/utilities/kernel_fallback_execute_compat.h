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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_H_

#include <functional>
#include <optional>
#include <string>

#include "status.h"
#include "threadpool_interface.h"
#include "types.h"
#include "kernel_fallback_compat_request_state.h"
#include "op_kernel_runner.h"
#include "op_attrs.h"  // from @tf_runtime
#include "async_value_ref.h"  // from @tf_runtime
#include "chain.h"  // from @tf_runtime
#include "execution_context.h"  // from @tf_runtime
#include "kernel_utils.h"  // from @tf_runtime
#include "forward_decls.h"  // from @tf_runtime
#include "tensor.h"  // from @tf_runtime

namespace tfrt {
class SyncKernelFrame;
}  // namespace tfrt

namespace tensorflow {
namespace tfd {

ABSL_CONST_INIT extern const char kOpKernelRunnerCacheResourceName[];

// The CoreRuntime dispatch function to run a TF kernel in kernel fallback
// compat mode.
tfrt::AsyncValueRef<tfrt::Chain> KernelFallbackExecuteCompatCoreRuntimeDispatch(
    const tfrt::ExecutionContext& exec_ctx, tfrt::string_view op_name,
    tfrt::string_view device_name, llvm::ArrayRef<tfrt::Tensor*> arguments,
    llvm::MutableArrayRef<tfrt::RCReference<tfrt::AsyncValue>> results,
    const KernelFallbackCompatRequestState& fallback_request_state,
    const tfrt_stub::OpKernelRunner& op_kernel_runner);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_EXECUTE_COMPAT_H_
