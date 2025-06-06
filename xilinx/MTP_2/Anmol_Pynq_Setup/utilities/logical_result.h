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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_LOGICAL_RESULT_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_LOGICAL_RESULT_H_

#include "LogicalResult.h"  // from @llvm-project

namespace xla {
namespace runtime {

using ::mlir::failure;  // NOLINT
using ::mlir::success;  // NOLINT

using ::mlir::failed;     // NOLINT
using ::mlir::succeeded;  // NOLINT

using ::mlir::FailureOr;      // NOLINT
using ::mlir::LogicalResult;  // NOLINT

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_LOGICAL_RESULT_H_
