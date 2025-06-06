/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "autotune_results.pb.h"
#include "hlo_module.h"
#include "executable.h"
#include "executable.pb.h"
#include "gpu_device_info.h"
#include "gpu_executable.h"
#include "hlo.pb.h"
#include "hlo_dataflow_analysis.h"
#include "hlo_pass_pipeline.h"
#include "llvm_compiler.h"
#include "statusor.h"
#include "device_description.pb.h"
#include "stream_executor.h"
#include "stream_executor_pimpl.h"
#include "util.h"

namespace xla {
namespace gpu {

// TODO(b/232263665): It should be shared between GPU and CPU.
class GpuXlaRuntimeAotCompilationResult : public AotCompilationResult {
 public:
  GpuXlaRuntimeAotCompilationResult(
      HloModuleProto hlo, std::string_view obj_file,
      std::string_view mlir_module, EntryFunctionAttributes entry_func_attrs,
      std::string_view gpu_asm_text, absl::Span<const uint8_t> gpu_binary,
      absl::Span<const GpuExecutable::ConstantInfo> constants = {}) {
    XlaRuntimeExecutableProto xla_runtime_executable;
    *xla_runtime_executable.mutable_hlo_module_proto() = hlo;
    xla_runtime_executable.set_obj_file(std::string(obj_file));
    xla_runtime_executable.set_mlir_module(std::string(mlir_module));
    *xla_runtime_gpu_executable_.mutable_xla_runtime_executable() =
        xla_runtime_executable;

    *xla_runtime_gpu_executable_.mutable_entry_func_attrs() = entry_func_attrs;
    xla_runtime_gpu_executable_.set_gpu_asm_text(std::string(gpu_asm_text));
    xla_runtime_gpu_executable_.set_gpu_binary(gpu_binary.data(),
                                               gpu_binary.size());

    for (const GpuExecutable::ConstantInfo& cst : constants) {
      auto* cst_proto = xla_runtime_gpu_executable_.add_constants();
      cst_proto->set_symbol_name(cst.symbol_name);
      cst_proto->set_allocation_index(cst.allocation_index);
      cst_proto->set_content(cst.content.data(), cst.content.size());
    }
  }

  explicit GpuXlaRuntimeAotCompilationResult(
      XlaRuntimeGpuExecutableProto executable)
      : xla_runtime_gpu_executable_(executable) {}

  StatusOr<std::string> SerializeAsString() const override {
    return xla_runtime_gpu_executable_.SerializeAsString();
  }

  static StatusOr<std::unique_ptr<GpuXlaRuntimeAotCompilationResult>>
  FromString(const std::string& serialized) {
    XlaRuntimeGpuExecutableProto xla_runtime_gpu_executable;
    if (!xla_runtime_gpu_executable.ParseFromString(serialized)) {
      return InternalError("Failed to parse serialized JitRtExecutableProto.");
    }
    return std::make_unique<GpuXlaRuntimeAotCompilationResult>(
        xla_runtime_gpu_executable);
  }

  StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, se::StreamExecutor* executor) const override;

 private:
  XlaRuntimeGpuExecutableProto xla_runtime_gpu_executable_;
};

struct GpuTargetConfig {
  GpuTargetConfig() = default;
  explicit GpuTargetConfig(const stream_executor::GpuTargetConfigProto& proto);

  se::GpuTargetConfigProto ToProto() const;

  GpuDeviceInfo gpu_device_info;
  GpuVersion gpu_version;
  std::string platform_name;
  se::dnn::VersionInfo dnn_version_info;
  std::string device_description_str;
};

// The GPU compiler generates efficient GPU executables.
class GpuCompiler : public LLVMCompiler {
 public:
  GpuCompiler(se::Platform::Id platform_id, const char* target_triple,
              const char* data_layout);

  using LLVMCompiler::Compile;

  // An attached device is passed in via stream_exec. We get GPU configuration
  // from the attached device. GemmAlgorithmPicker and GpuConvAlgorithmPicker
  // can run on the attached device.
  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  // Run HloPasses without an attached deivce. So GemmAlgorithmPicker and
  // GpuConvAlgorithmPicker can not run.
  StatusOr<std::unique_ptr<HloModule>> RunHloPassesWithoutDevice(
      std::unique_ptr<HloModule> module, const CompileOptions& options,
      const GpuTargetConfig& gpu_target_config,
      const AutotuneResults& autotune_results);

  StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(
      HloModule* hlo_module, se::StreamExecutor* stream_exec) override;

  virtual GpuVersion GetGpuVersion(se::StreamExecutor* stream_exec) = 0;
  GpuTargetConfig GetGpuTargetConfig(se::StreamExecutor* stream_exec) {
    GpuTargetConfig gpu_target_config;
    gpu_target_config.gpu_device_info = GetGpuDeviceInfo(stream_exec);
    gpu_target_config.gpu_version = GetGpuVersion(stream_exec);
    gpu_target_config.platform_name = stream_exec->platform()->Name();

    return gpu_target_config;
  }

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     AotCompilationOptions const& options) override;

  StatusOr<std::pair<std::string, std::vector<uint8_t>>> CompileToTargetBinary(
      const HloModuleConfig& module_config,
      std::unique_ptr<llvm::Module> llvm_module, GpuVersion gpu_version,
      se::StreamExecutor* stream_exec, const CompileOptions& options,
      const HloModule* debug_module);

  se::Platform::Id PlatformId() const override { return platform_id_; }

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  // Returns a (deserialized) AotCompilationResult from a serialized
  // AotCompilationResult.
  StatusOr<std::unique_ptr<AotCompilationResult>> LoadAotCompilationResult(
      const std::string& serialized_aot_result) override {
    return GpuXlaRuntimeAotCompilationResult::FromString(serialized_aot_result);
  }

  StatusOr<std::unique_ptr<AotCompilationResult>> Export(
      Executable* executable) const override;

  static std::optional<bool> FusionCanShareBufferHint(
      const HloInstruction* user, const HloInstruction* operand,
      const ShapeIndex& user_index);

 protected:
  // During compilation with device, stream_exec != null and autotune_results
  // == null. During deviceless AOT compilation, stream_exec == null and
  // autotune_results != null.
  virtual Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options, const GpuTargetConfig& gpu_target_config,
      const AutotuneResults* autotune_results);

  // Linearize collective schedule under SPMD partitioning if online autotuning
  // of convolutions is enabled.
  virtual bool EnableCollectiveScheduleLinearizerForSpmd(
      HloModule* hlo_module, se::StreamExecutor* stream_exec) {
    return false;
  }

  // CollectivesScheduleLinearizer enforces a total ordering between collectives
  // to work around (1) divergence in initial HLOs across executables that are
  // communicating with each other using HLO collectives, and (2) divergence in
  // executables introduced due to auto tuning, specifically the use of extra
  // scratch space for convolutions.
  // We always apply this pass when not using SPMD (where initial HLO divergence
  // may be possible). This function decided whether to apply this pass when
  // using SPMD partitioning. When using SPMD, if convolutions are present in
  // the code and we are using "online" autotuning (i.e., not AOT) we need to
  // use the pass, else we do not need to enable the pass.
  virtual bool RequiresCollectiveScheduleLinearizer(const HloModule* module) {
    return false;
  }

  // Add autotuning passes for convolution, gemm and triton.
  virtual Status AddAutotuningPasses(HloPassPipeline* pipeline,
                                     HloModule* hlo_module,
                                     se::StreamExecutor* stream_exec,
                                     const DebugOptions& debug_options,
                                     const CompileOptions& options,
                                     const GpuTargetConfig& gpu_target_config,
                                     const AutotuneResults* autotune_results,
                                     tsl::thread::ThreadPool* thread_pool) {
    return OkStatus();
  }

  virtual Status LoadAutotuneResultsFromFile(
      const DebugOptions& debug_options) {
    return OkStatus();
  }

  virtual Status SerializeAutotuneResultsToFile(
      const DebugOptions& debug_options) {
    return OkStatus();
  }

 private:
  // During compilation with device, stream_exec != null and autotune_results
  // == null. During deviceless AOT compilation, stream_exec == null and
  // autotune_results != null.
  Status OptimizeHloModule(HloModule* hlo_module,
                           se::StreamExecutor* stream_exec,
                           const CompileOptions& options,
                           const GpuTargetConfig& gpu_target_config,
                           const AutotuneResults* autotune_results);

  virtual Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, GpuVersion gpu_version,
      se::dnn::VersionInfo dnn_version,
      se::DeviceMemoryAllocator* device_allocator) = 0;

  virtual HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer() {
    return &FusionCanShareBufferHint;
  }

  // TODO(timshen): Replace `debug_module` with some portable debug information
  // that accommodates both HLO and MLIR.
  virtual StatusOr<std::pair<std::string, std::vector<uint8_t>>>
  CompileTargetBinary(const HloModuleConfig& module_config,
                      llvm::Module* llvm_module, GpuVersion gpu_version,
                      bool relocatable, const HloModule* debug_module,
                      const CompileOptions& options) = 0;

  Status PrepareHloModuleForIrEmitting(HloModule* hlo_module);

  virtual StatusOr<bool> CanUseLinkModules(const HloModuleConfig& config) {
    return false;
  }

  virtual StatusOr<std::vector<uint8_t>> LinkModules(
      se::StreamExecutor* stream_exec,
      std::vector<std::vector<uint8_t>> modules,
      const DebugOptions& debug_options) {
    return Unimplemented("LinkModules is not implemented.");
  }

  se::Platform::Id platform_id_;

  // The triple that represents our target.
  const char* target_triple_;

  // The data layout of the emitted module.
  const char* data_layout_;

  // The size in bytes of a pointer. Used by ShapeSizeBytesFunction.
  const int64_t pointer_size_;

  GpuCompiler(const GpuCompiler&) = delete;
  GpuCompiler& operator=(const GpuCompiler&) = delete;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
