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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_COMPILER_OPTIONS_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_XLA_COMPILER_OPTIONS_UTIL_H_

#include "device_compiler.h"
#include "xla_platform_info.h"
#include "xla_compiler.h"
#include "local_client.h"
#include "pjrt_client.h"

namespace tensorflow {

// Returns created options for the XLA compiler.
XlaCompiler::Options GenerateCompilerOptions(
    const DeviceCompiler<xla::LocalExecutable, xla::LocalClient>&
        xla_device_compiler,
    const FunctionLibraryRuntime& function_library, DeviceBase* device,
    se::Stream* stream, const XlaPlatformInfo& platform_info,
    bool has_ref_vars);

// Returns created options for XLA compiler when TFRT-TPU is used.
XlaCompiler::Options GenerateCompilerOptionsForTfrtTpu(
    const DeviceCompiler<xla::LocalExecutable, xla::LocalClient>&
        xla_device_compiler,
    const FunctionLibraryRuntime& function_library);

// Returns created options for XLA compiler when PjRt (Device API) is used for
// compilation and execution.
// TODO(b/255826209): Remove default arg once PjRtCompileOnDemand op is deleted.
XlaCompiler::Options GenerateCompilerOptionsForPjRt(
    const FunctionLibraryRuntime& function_library,
    const DeviceBase* device_base, const XlaPlatformInfo& platform_info,
    const DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>*
        pjrt_device_compiler = nullptr);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_COMPILER_OPTIONS_UTIL_H_
