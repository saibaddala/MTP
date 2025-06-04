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

// This file defines the operations used in the LHLO dialect.

#ifndef MLIR_HLO_LHLO_IR_LHLO_OPS_H
#define MLIR_HLO_LHLO_IR_LHLO_OPS_H

#include "lhlo_ops_structs.h"
#include "lhlo_structured_interface.h"
#include "StringRef.h"
#include "hlo_ops.h"
#include "Bufferization.h"
#include "FuncOps.h"
#include "MemRef.h"
#include "Attributes.h"
#include "BuiltinTypes.h"
#include "Dialect.h"
#include "Location.h"
#include "MLIRContext.h"
#include "OpDefinition.h"
#include "Operation.h"
#include "Types.h"
#include "ControlFlowInterfaces.h"
#include "CopyOpInterface.h"
#include "LoopLikeInterface.h"
#include "SideEffectInterfaces.h"
#include "ViewLikeInterface.h"

namespace mlir {
class OpBuilder;
namespace lmhlo {

class LmhloDialect : public Dialect {
 public:
  explicit LmhloDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "lmhlo"; }

  // Parses an attribute registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  // Prints an attribute registered to this dialect.
  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;
};

}  // namespace lmhlo
}  // end namespace mlir

#define GET_OP_CLASSES
#include "lhlo_ops.h.inc"

#endif  // MLIR_HLO_LHLO_IR_LHLO_OPS_H
