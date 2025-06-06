/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_IR_MLRT_TF_MLRT_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_IR_MLRT_TF_MLRT_OPS_H_

#include "BuiltinOps.h"  // from @llvm-project
#include "Dialect.h"  // from @llvm-project
#include "OpDefinition.h"  // from @llvm-project
#include "OpImplementation.h"  // from @llvm-project
#include "SideEffectInterfaces.h"  // from @llvm-project
#include "tf_side_effects.h"
#include "tfrt_op_interfaces.h"  // from @tf_runtime
#include "tfrt_traits.h"  // from @tf_runtime

namespace tensorflow {
namespace tf_mlrt {

class TensorflowMlrtDialect : public mlir::Dialect {
 public:
  explicit TensorflowMlrtDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "tf_mlrt"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type, mlir::DialectAsmPrinter &os) const override;
};

// The MLIR type represents a tensorflow::Tensor.
class TFTensorType
    : public mlir::Type::TypeBase<TFTensorType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
};

// The MLIR type represents a tensorflow::Device*
class TFDeviceType
    : public mlir::Type::TypeBase<TFDeviceType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace tf_mlrt
}  // namespace tensorflow

#define GET_OP_CLASSES
#include "tf_mlrt_ops.h.inc"
#define GET_OP_CLASSES
#include "tf_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_IR_MLRT_TF_MLRT_OPS_H_
