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

#ifndef TENSORFLOW_DTENSOR_MLIR_SPMD_EXPANDER_COMMON_H_
#define TENSORFLOW_DTENSOR_MLIR_SPMD_EXPANDER_COMMON_H_

#include <optional>
#include <string>
#include <vector>

#include "string_view.h"
#include "ArrayRef.h"
#include "SmallPtrSet.h"
#include "SmallVector.h"
#include "Casting.h"
#include "FuncOps.h"  // from @llvm-project
#include "Builders.h"  // from @llvm-project
#include "BuiltinOps.h"  // from @llvm-project
#include "BuiltinTypes.h"  // from @llvm-project
#include "MLIRContext.h"  // from @llvm-project
#include "Value.h"  // from @llvm-project
#include "tf_device.h"
#include "tf_ops.h"
#include "tf_remaining_ops.h"
#include "dstatus.h"
#include "tensor_layout.h"
#include "tf_dtensor.h"

namespace tensorflow {
namespace dtensor {

constexpr absl::string_view kReduceOpAdd = "Add";
constexpr absl::string_view kReduceOpAll = "All";
constexpr absl::string_view kReduceOpAny = "Any";
constexpr absl::string_view kReduceOpMax = "Max";
constexpr absl::string_view kReduceOpMin = "Min";
constexpr absl::string_view kReduceOpMul = "Mul";
// Mean is not a valid combinator function on its own. It is handled specially
// by the reduce expansion.
constexpr absl::string_view kReduceOpMean = "Mean";

// Returns true if all layouts are replicated.
bool AllReplicated(const std::vector<Layout>& layouts);

// Takes a global type and converts it to a local type. Fails if the number of
// shards does not divide the size of the dimension (if not dynamic).
StatusOr<mlir::TensorType> LocalTypeFromGlobalType(
    const Layout& layout, const mlir::TensorType& original_type);

// Takes a global type and converts it to a local type.
StatusOr<mlir::TensorType> GlobalTypeFromLocalType(
    const Layout& layout, const mlir::TensorType& original_type);

// Creates a tf::SplitOp that splits 'src_input' into 'num_splits' ways
// in 'split_dimension' dimension and returns the split values.
Status CreateSplitOp(int num_split, int split_dimension,
                     mlir::Location location, mlir::Value src_input,
                     mlir::OpBuilder* builder, mlir::TF::SplitOp* split_op);

// Given layouts + shapes, determines if the two are broadcast compatible.
// See source file for more documentation.
StatusOr<Layout> GetBroadcastLayoutForElementWise(
    const Layout& layout_a, const Layout& layout_b,
    mlir::ArrayRef<int64_t> shape_a, mlir::ArrayRef<int64_t> shape_b,
    int64_t dims_to_ignore, std::vector<std::string>& to_split_a,
    std::vector<std::string>& to_split_b);

// Returns a merged layout using `GetBroadcastLayoutForElementwise()` function
// given a list of operand layouts.
StatusOr<std::optional<Layout>> GetMergedOperandLayout(
    const llvm::DenseMap<int, Layout>& operand_layouts, mlir::Operation* op);

// Returns the forwarded input value of DTensorLayout op for which `value` is
// the output. This must be used after layout propagation and before SPMD
// expansion when all mlir::Value's of tf ops are followed by DTensorLayout op
// to specify output layout.
// To make the implementation safe for Layout Propagation V1 algorithm, if the
// defining op of `value` is not DTensorLayout op (only the case for V1),
// returns `value` directly.
// TODO(b/172936130): Remove special casing for v1 Layout Propagation
// algorithm.
mlir::Value GetForwardedDTensorLayoutInput(mlir::Value value);

// Goal of this function is to connect 'mlir::Value's (read 'mlir::OpResult's)
// to the 'mlir::OpOperand's which use them, crossing function call
// boundaries. The only keys in consumers which will not actually be
// 'mlir::OpResult's will be the 'mlir::Value's representing the inputs of the
// main function. The rest will be direct output of operations -- i.e.
// mlir::OpResult. Note that 'mlir::Value's that are not used by any op or are
// simply returned from the main functiuon will not be in this list. In these
// cases, there are no conditions on the layouts for these 'mlir::Value's.
//
// A list of current assumptions in this code:
// * Functions are only called once.
// * Functions that are not reachable from main have been trimmed.
// * Input to CopyToMesh can always be traced back to function inputs.
mlir::LogicalResult PopulateConsumersFromModule(
    mlir::ModuleOp* module, mlir::Dialect* tf_dialect,
    llvm::DenseMap<mlir::Value, std::vector<mlir::OpOperand*>>& consumers);

// From device id, return an mlir::Value for a tensor of shape [1,
// mesh.rank()] whose entries are the mesh coordinates of the device. The mesh
// used, is the mesh for the given cluster.
StatusOr<mlir::Value> GetMeshCoordinatesFromCluster(
    mlir::tf_device::ClusterOp cluster);

// Returns Mesh attribute on the parent cluster op for the input operation.
StatusOr<Mesh> GetMeshOnParentCluster(mlir::Operation* op);

// Checks that optional metadata attributes of `op` are valid if they
// exist. More specifically, output layouts of tf.Shape op and layouts of
// resources inferred from AssignVariable op is added as metadata.
mlir::LogicalResult ValidateMetadataAttributes(mlir::Operation* op);

// Creates a map from function to ops which calls the function.
mlir::LogicalResult GetFuncToCaller(
    mlir::ModuleOp module,
    llvm::DenseMap<llvm::StringRef, mlir::Operation*>& func_to_caller);

// Takes an operand and traces its use across function call and
// tf_device.cluster boundaries. Note that this may turn one operand into
// many.
llvm::SmallVector<mlir::OpOperand*, 4> TraceUseToNextTFOp(
    mlir::OpOperand* operand,
    const llvm::DenseMap<llvm::StringRef, mlir::Operation*>& func_to_caller,
    llvm::SmallVector<mlir::Value, 4>* skipped_values = nullptr);

// Replaces `cluster` with a new tf_device.cluster without return values
// if result values are not used by any other ops.
//
// For example:
//
//  %unused_value  = "tf_device.cluster"() ({
//      %1 = "tf.Const"() {value = dense<10> : tensor<i32>} : () ->
//      tensor<i32> %2 = "tf.Neg"(%1) : (tensor<i32>) -> tensor<i32>
//      tf_device.return %2 : tensor<i32>
//  }) {_mesh="mesh:CPU,x=2,y=2"} : () -> (tensor<i32>)
//
// Will be transformed to:
//
//  "tf_device.cluster"() ({
//      %1 = "tf.Const"() {value = dense<10> : tensor<i32>} : () ->
//      tensor<i32> %2 = "tf.Neg"(%1) : (tensor<i32>) -> tensor<i32>
//      tf_device.return
//  }) {_mesh="mesh:CPU,x=2,y=2"} : () -> ()
void RemoveUnusedClusterResults(mlir::tf_device::ClusterOp cluster);

mlir::StringAttr GetUniqueControlflowFnName(const std::string& prefix,
                                            mlir::OpBuilder& builder);

// Sets the builder insertion point to after value. If value is a block
// argument, this checks that all users of the value are in the same cluster.
// If not it errors out. If they are then it sets the inserition point to the
// top of the cluster.
Status SetBuilderInsertionAfterValue(mlir::Value value,
                                     mlir::OpBuilder& builder);

// Inserts a StringFormat and Print op, should only be used for debugging
// on CPU.
Status PrintTensor(mlir::Value value, const std::string& format_string);

// Extract a vector of string from mlir value.
Status ExtractConstStringVectorFromValue(
    mlir::Value value, llvm::SmallVectorImpl<std::string>& out_vector);

StatusOr<std::string> ExtractConstScalarStringFromValue(mlir::Value value);

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_SPMD_EXPANDER_COMMON_H_
