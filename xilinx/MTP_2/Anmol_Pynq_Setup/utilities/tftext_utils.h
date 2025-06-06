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

// This header file defines common utils used by TFLite transformation
// passes to work with op attributes.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_TFTEXT_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_TFTEXT_UTILS_H_

#include "StringRef.h"
#include "FuncOps.h"  // from @llvm-project
#include "Builders.h"  // from @llvm-project
#include "BuiltinOps.h"  // from @llvm-project
#include "BuiltinTypes.h"  // from @llvm-project
#include "Location.h"  // from @llvm-project
#include "Value.h"  // from @llvm-project
#include "LogicalResult.h"  // from @llvm-project
#include "tfl_ops.h"
#include "tf_attributes.h"
#include "op.h"

namespace mlir {
namespace TFL {

// Fuse TF.Text APIs annotated by tf.function to a TFLite custom op.
LogicalResult ConvertTFTextAPI(mlir::func::FuncOp func, llvm::StringRef api,
                               mlir::TF::FuncAttr attr);

// Check if TF.Text Tensorflow ops are registered.
bool IsTFTextRegistered(const tensorflow::OpRegistry* op_registery);

}  // end namespace TFL
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_TFTEXT_UTILS_H_
