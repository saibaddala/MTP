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

// This file defines the operations and types used in the XLAFramework dialect.
//
#ifndef TENSORFLOW_COMPILER_XLA_MLIR_XLA_CPU_IR_XLA_CPU_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_XLA_CPU_IR_XLA_CPU_H_

#include "BufferizableOpInterface.h"  // from @llvm-project
#include "Attributes.h"  // from @llvm-project
#include "BuiltinTypes.h"  // from @llvm-project
#include "Dialect.h"  // from @llvm-project
#include "MLIRContext.h"  // from @llvm-project
#include "OpDefinition.h"  // from @llvm-project
#include "OpImplementation.h"  // from @llvm-project
#include "InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "xla_cpu.h.inc"
#include "xla_cpu_dialect.h.inc"
#include "xla_cpu_enums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "xla_cpu_attrdefs.h.inc"
#undef GET_OP_CLASSES

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_XLA_CPU_IR_XLA_CPU_H_
