#include "constants.h"
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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_H_

#include "OpDefinition.h"  // from @llvm-project
#include "Operation.h"  // from @llvm-project
#include "SymbolTable.h"  // from @llvm-project
#include "LogicalResult.h"  // from @llvm-project
#include "sharding_param.h"

namespace mlir {
namespace OpTrait {
namespace xla {
namespace ifrt {

namespace impl {

// Verifies `op` used in a FuncOp with `ifrt.function` attr.
LogicalResult verifyNestedInIfrtFunc(Operation* op);

}  // namespace impl

template <typename ConcreteType>
class NestedInIfrtFuncTrait
    : public TraitBase<ConcreteType, NestedInIfrtFuncTrait> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return impl::verifyNestedInIfrtFunc(op);
  }
};

template <typename CalleeOpType>
class IfrtCallLikeTrait {
 public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, Impl> {
   public:
    // Verifies getCallee() is a valid SymbolRefAttr to CalleeOpType.
    static LogicalResult verifyTrait(Operation* op) {
      mlir::SymbolTableCollection symbol_table;
      ConcreteType concrete = llvm::cast<ConcreteType>(op);
      CalleeOpType callee = concrete.getCalleeOp(symbol_table);
      if (callee == nullptr) {
        return op->emitOpError() << "requires '" << concrete.getCallee()
                                 << "' to reference a valid `"
                                 << CalleeOpType::getOperationName() << "`";
      }
      if (callee->hasAttr(::xla::ifrt::kIfrtFunctionAttrName)) {
        return op->emitOpError() << "requires callee not with attr `"
                                 << ::xla::ifrt::kIfrtFunctionAttrName << "`";
      }
      return success();
    }

    CalleeOpType getCalleeOp(mlir::SymbolTableCollection& symbol_table) {
      SymbolRefAttr callee_attr = static_cast<ConcreteType*>(this)->getCallee();
      return symbol_table.lookupNearestSymbolFrom<CalleeOpType>(
          this->getOperation(), callee_attr);
    }
  };
};

}  // namespace ifrt
}  // namespace xla
}  // namespace OpTrait
}  // namespace mlir

// Generated definitions.
#define GET_OP_INTERFACE_CLASSES
#include "ifrt_interfaces.h.inc"  // IWYU pragma: export

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_H_
