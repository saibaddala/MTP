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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_OPS_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_OPS_H_

#include "Casting.h"
#include "FuncOps.h"  // from @llvm-project
#include "BuiltinAttributes.h"  // from @llvm-project
#include "BuiltinTypes.h"  // from @llvm-project
#include "OpDefinition.h"  // from @llvm-project
#include "OpImplementation.h"  // from @llvm-project
#include "Operation.h"  // from @llvm-project
#include "ValueRange.h"  // from @llvm-project
#include "Visitors.h"  // from @llvm-project
#include "CallInterfaces.h"  // from @llvm-project
#include "LogicalResult.h"  // from @llvm-project
#include "constants.h"
#include "ifrt_dialect.h"
#include "ifrt_interfaces.h"

// Generated definitions.
#define GET_OP_CLASSES
#include "ifrt_ops.h.inc"  // IWYU pragma: export

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_OPS_H_
