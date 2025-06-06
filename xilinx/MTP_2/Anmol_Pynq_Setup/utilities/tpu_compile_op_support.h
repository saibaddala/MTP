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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_SUPPORT_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_SUPPORT_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "string_view.h"
#include "span.h"
#include "ops.h"
#include "xla_compiler.h"
#include "hlo_module_group.h"
#include "hlo_sharding.h"
#include "hlo_module_config.h"
#include "shape.h"
#include "shape_tree.h"
#include "statusor.h"
#include "xla_data.pb.h"
#include "attr_value.pb.h"
#include "function.h"
#include "op_kernel.h"
#include "tensor.pb.h"
#include "tensor_shape.h"
#include "types.pb.h"
#include "compile_metadata.pb.h"
#include "tpu_compile.pb.h"

namespace tensorflow {
namespace tpu {

namespace se = ::stream_executor;

// List of parameters for lowering Mlir to HLO IR.
struct MlirToHloArgs {
  absl::string_view mlir_module;
  ConfigProto::Experimental::MlirBridgeRollout rollout_state =
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;
};

// Variant of guaranteed constant tensors types.
using GuaranteedConsts = std::variant<absl::Span<const TensorProto* const>,
                                      const OpInputList* const>;

// List of parameters for lowering function library definition to HLO IR.
struct FunctionToHloArgs {
  const NameAttrList* const function;
  const FunctionLibraryDefinition* const flib_def;
  int graph_def_version;
  GuaranteedConsts guaranteed_constants;
};

// Persistent cache for compiled TPU program and the related compiler metadata
// intended for TPU inference.
// TODO(henrytan): there is an opportunity to consolidate the interface with the
// `TpuCompilationCacheInterface` once `TpuPersistentCompilationCache` is
// converted into a ref count based class.
class TpuPersistentCompilationCacheInterface {
 public:
  virtual ~TpuPersistentCompilationCacheInterface() = default;

  // Returns the location where cache entries are stored.
  virtual std::string cache_location() const = 0;
};

// Describes the position of an argument or return value after the computation
// has been partitioned into cores.
struct ShardingAndIndex {
  // Sharding across cores.
  ::xla::OpSharding sharding;
  // Argument/return value number. If sharding is single-core, `indices` has a
  // single element; otherwise, it has num_cores elements.
  std::vector<int> indices;
};

// TODO(b/158279168): Dedup with internal version.
// Return the per-device shape for a `shape` with a given `sharding`.
xla::Shape GetPerDeviceShape(const xla::Shape& shape,
                             const xla::HloSharding& sharding, int64_t device);

tsl::StatusOr<std::unique_ptr<xla::HloModuleConfig>> CreateModuleConfig(
    const xla::ProgramShape& program_shape,
    absl::Span<const xla::Shape> argument_shapes,
    std::optional<const xla::Shape> result_layout,
    std::optional<const xla::DeviceAssignment> device_assignment,
    int replica_count, int num_partitions,
    const xla::DebugOptions* debug_options, const int* seed,
    const int* launch_id, const bool* alias_passthrough_params,
    const xla::FusionConfigCollection* fusion_config_collection,
    const std::vector<std::vector<bool>>* fusion_config);

tsl::StatusOr<std::unique_ptr<xla::HloModuleConfig>> CreateModuleConfig(
    const xla::ProgramShape& program_shape,
    absl::Span<const xla::Shape> argument_shapes,
    std::optional<const xla::Shape> result_layout,
    std::optional<const xla::DeviceAssignment> device_assignment,
    int replica_count, int num_partitions,
    const xla::DebugOptions* debug_options);

xla::ShapeTree<xla::HloSharding> GetSubtree(
    const xla::ShapeTree<xla::HloSharding>& tuple_shape_tree,
    int element_index);

xla::Shape GetPerDeviceShape(const xla::Shape& shape,
                             const xla::HloSharding& sharding, int64_t device);

Status AddVariableUpdatesToCores(
    const TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    const std::vector<ShardingAndIndex>& arg_core_mapping,
    std::vector<bool>* may_modify_variables,
    std::vector<std::vector<xla::Shape>>* per_core_output_shapes,
    std::vector<std::vector<std::pair<int, bool>>>* per_core_variable_indices);

tsl::Status ComputeOutputShapesForEachCore(
    const tpu::TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    std::vector<std::vector<xla::Shape>>* per_core_output_shapes);

tsl::Status CreateHloModules(
    const TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    const std::optional<xla::DeviceAssignment>& device_assignment,
    std::vector<std::unique_ptr<xla::HloModule>>* hlo_modules);

tsl::StatusOr<TpuCompilationRequestProto> CreateTpuCompilationRequest(
    const std::variant<MlirToHloArgs, FunctionToHloArgs>& computation,
    const TPUCompileMetadataProto& metadata,
    const std::vector<TensorShape>& arg_shapes);

tsl::Status CompileOpMetadataFromContext(OpKernelConstruction* ctx,
                                         TPUCompileMetadataProto* metadata,
                                         NameAttrList* function_name,
                                         std::string* mlir_module);

// Computes shapes for each argument. Uses both the static shape from the
// metadata, and the dynamic shapes where the static shape is not
// defined. There must be one dynamic_shape for each argument with a
// partially defined shape, in index order.
Status ComputeArgumentShapes(const TPUCompileMetadataProto& metadata,
                             const std::vector<TensorShape>& dynamic_shapes,
                             std::vector<TensorShape>* arg_shapes);
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_SUPPORT_H_
