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

// This file defines the tf_executor dialect: it models the TensorFlow executor
// semantics and can represent arbitrary TensorFlow graphs. As such it follows
// the existing execution model that includes deadness propagation, concurrent
// semantics, and control dependencies.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_EXECUTOR_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_EXECUTOR_H_

#include "Traits.h"  // from @llvm-project
#include "Attributes.h"  // from @llvm-project
#include "Builders.h"  // from @llvm-project
#include "BuiltinTypes.h"  // from @llvm-project
#include "Dialect.h"  // from @llvm-project
#include "Matchers.h"  // from @llvm-project
#include "OpImplementation.h"  // from @llvm-project
#include "InferTypeOpInterface.h"  // from @llvm-project
#include "tf_types.h"

namespace mlir {
namespace tf_executor {

class TensorFlowExecutorDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tf_executor"; }
  explicit TensorFlowExecutorDialect(MLIRContext *context);

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;
};

// The Control type is a token-like value that models control dependencies from
// TensorFlow graphs.
class ControlType : public Type::TypeBase<ControlType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class TokenType : public Type::TypeBase<TokenType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace tf_executor
}  // namespace mlir

// Declares the operations for this dialect using the generated header.
#define GET_OP_CLASSES
#include "tf_executor.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_EXECUTOR_H_
