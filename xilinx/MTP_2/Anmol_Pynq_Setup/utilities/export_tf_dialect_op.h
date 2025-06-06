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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_EXPORT_TF_DIALECT_OP_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_EXPORT_TF_DIALECT_OP_H_

#include "StringRef.h"
#include "Operation.h"  // from @llvm-project
#include "export_utils.h"
#include "attr_value.pb.h"
#include "node_def.pb.h"
#include "node_def_util.h"
#include "op_def_builder.h"
#include "status.h"

namespace tensorflow {

// Extracts the attributes of a MLIR operation and populates the converted
// attributes in a proto map<string, AttrValue>.
Status GetAttrValuesFromOperation(
    mlir::Operation* inst, llvm::StringRef name,
    const tensorflow::OpRegistrationData* op_reg_data,
    bool ignore_unregistered_attrs, AttrValueMap* attributes);

// Converts a MLIR operation to TensorFlow NodeDef with given node name. This
// name should be unique to the graph it is being inserted to. If the
// `ignore_unregistered_attrs` argument is set to true, the attributes which are
// not in the op registry will be ignored. If the `ignore_unregistered_attrs`
// argument is not set to true, _output_shapes attribute is added to nodes with
// ShapedType for the leading values with ShapedType in the results of the
// nodes. Set it to true if the returned NodeDef will be executed by the linked
// TF Eager runtime.
StatusOr<std::unique_ptr<NodeDef>> ConvertTFDialectOpToNodeDef(
    mlir::Operation* inst, llvm::StringRef name,
    bool ignore_unregistered_attrs);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_EXPORT_TF_DIALECT_OP_H_
