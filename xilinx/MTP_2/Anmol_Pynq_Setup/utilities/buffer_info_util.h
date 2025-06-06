/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_BUFFER_INFO_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_BUFFER_INFO_UTIL_H_

#include "span.h"
#include "cpu_function_runtime.h"
#include "hlo_module.h"
#include "buffer_assignment.h"

namespace xla {
namespace cpu {
// Creates and returns a list of BufferInfo instances containing relevant
// information from `buffer_assignment`.
std::vector<cpu_function_runtime::BufferInfo>
CreateBufferInfosFromBufferAssignment(
    const HloModule& module, const BufferAssignment& buffer_assignment);

// Creates and returns a table containing the mapping from entry computation
// parameters to buffer allocation indices.
//
// If this function returns V then entry parameter i has buffer allocation index
// V[i].
std::vector<int32_t> CreateArgIndexTableFromBufferInfos(
    absl::Span<const cpu_function_runtime::BufferInfo> buffer_infos);

std::vector<int32_t> CreateResultIndexTableFromBufferInfos(
    absl::Span<const cpu_function_runtime::BufferInfo> buffer_infos);
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_BUFFER_INFO_UTIL_H_
