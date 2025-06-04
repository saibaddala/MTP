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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_IR_RT_OPS_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_IR_RT_OPS_H_

#include "FuncOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "BuiltinOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "SymbolTable.h"  // from @llvm-project  // IWYU pragma: keep
#include "ControlFlowInterfaces.h"  // from @llvm-project  // IWYU pragma: keep
#include "SideEffectInterfaces.h"  // from @llvm-project  // IWYU pragma: keep
#include "rt_dialect.h"  // IWYU pragma: keep
#include "rt_interfaces.h"  // IWYU pragma: keep

#define GET_OP_CLASSES
#include "rt_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_IR_RT_OPS_H_
