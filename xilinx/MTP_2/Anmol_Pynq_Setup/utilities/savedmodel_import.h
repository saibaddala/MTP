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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_SAVEDMODEL_IMPORT_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_SAVEDMODEL_IMPORT_H_

#include "BuiltinOps.h"  // from @llvm-project
#include "MLIRContext.h"  // from @llvm-project
#include "OwningOpRef.h"  // from @llvm-project
#include "graph_debug_info.pb.h"
#include "statusor.h"
#include "saved_model.pb.h"

namespace mlir {
namespace tfg {

// Converts a saved model to a MLIR module expressed in TFG dialect.
// Only the root graph and function library of the saved model gets imported
// into MLIR TFG dialect.
// TODO(b/218882780): Consider importing SignatureDefs from the SavedModel.
tensorflow::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportSavedModelToMlir(
    mlir::MLIRContext* context, const tensorflow::GraphDebugInfo& debug_info,
    const tensorflow::SavedModel& saved_model);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_SAVEDMODEL_IMPORT_H_
