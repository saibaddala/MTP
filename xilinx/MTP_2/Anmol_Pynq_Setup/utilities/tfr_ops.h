/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_OPS_H_

#include "StringSet.h"
#include "FuncOps.h"  // from @llvm-project
#include "Shape.h"  // from @llvm-project
#include "BuiltinOps.h"  // from @llvm-project
#include "BuiltinTypes.h"  // from @llvm-project
#include "Dialect.h"  // from @llvm-project
#include "DialectImplementation.h"  // from @llvm-project
#include "FunctionInterfaces.h"  // from @llvm-project
#include "MLIRContext.h"  // from @llvm-project
#include "Types.h"  // from @llvm-project
#include "CallInterfaces.h"  // from @llvm-project
#include "ControlFlowInterfaces.h"  // from @llvm-project
#include "InferTypeOpInterface.h"  // from @llvm-project
#include "SideEffectInterfaces.h"  // from @llvm-project

namespace mlir {
namespace TFR {

constexpr char kAttrArgumentNameAttr[] = "tfr.name";
constexpr char kAttrArgumentDefaultAttr[] = "tfr.default";
constexpr char kAttrArgumentTypeAttr[] = "tfr.type";

class TFRDialect : public Dialect {
 public:
  explicit TFRDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "tfr"; }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  // Parse a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type ty, DialectAsmPrinter &os) const override;
};

}  // namespace TFR
}  // namespace mlir

#define GET_OP_CLASSES
#include "tfr_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_OPS_H_
