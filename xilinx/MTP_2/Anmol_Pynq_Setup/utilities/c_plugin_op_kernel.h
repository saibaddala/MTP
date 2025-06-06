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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_OP_KERNEL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_OP_KERNEL_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernels.h"
#include "pjrt_client.h"
#include "plugin_coordination_service_agent.h"
#include "plugin_op_kernel.h"
#include "function.h"
#include "config.pb.h"
#include "thread_annotations.h"

namespace tensorflow {

class CPluginOpKernelConstruction : public PluginOpKernelConstruction {
 public:
  explicit CPluginOpKernelConstruction(void* ctx)
      : ctx_(reinterpret_cast<TF_OpKernelConstruction*>(ctx)) {}

  Status GetBoolAttr(std::string_view attr_name, bool* value) const override;
  Status GetInt32Attr(std::string_view attr_name, int* value) const override;
  Status GetInt32AttrList(std::string_view attr_name,
                          std::vector<int32_t>* value) const override;
  Status GetInt64Attr(std::string_view attr_name,
                      int64_t* value) const override;
  Status GetStringAttr(std::string_view attr_name,
                       std::string* value) const override;
  Status GetFunctionAttr(std::string_view attr_name,
                         NameAttrList* function) const override;

  void CtxFailure(const Status& status) override;
  void CtxFailure(const char* file, int line, const Status& status) override;

  void* GetContext() const override { return ctx_; }

 private:
  TF_OpKernelConstruction* ctx_;  // not owned.
};

class CPluginOpKernelContext : public PluginOpKernelContext {
 public:
  explicit CPluginOpKernelContext(void* ctx)
      : ctx_(reinterpret_cast<TF_OpKernelContext*>(ctx)) {}

  std::string_view GetResourceMgrDefaultContainerName() override;

  Status LookupOrCreateResource(std::string_view container_name,
                                std::string_view plugin_resource_name,
                                void** result_plugin_resource,
                                void* (*create_func)(void*),
                                void* create_func_args,
                                void (*delete_func)(void*)) override;

  PluginCoordinationServiceAgent* GetPluginCoordinationServiceAgent()
      const override;

  Status CreatePluginVariable(int index,
                              PluginVariable** variable) const override;

  Status AllocateTempForPluginVariable(PluginVariable* variable) override;

  int NumInputs() const override { return TF_NumInputs(ctx_); }

  Status GetInput(int index, Tensor* tensor) const override;

  Status GetInput(const char* name, const Tensor** tensor) override;

  Status GetInputRange(std::string_view name,
                       std::pair<int, int>* range) const override;

  DataType GetInputDataType(int index) const override;

  std::string_view GetOpKernelRequestedInput(int index) const override;

  std::string_view GetOpKernelName() const override;

  uint64_t GetFrameId() const override { return TF_GetFrameId(ctx_); }

  int64_t GetIterId() const override { return TF_GetIterId(ctx_); }

  int64_t GetStepId() const override { return TF_GetStepId(ctx_); }

  int GetDeviceId() const override { return TF_GetDeviceId(ctx_); }

  std::string GetSessionName() const override {
    // TODO(haoyuzhang): Implement with ctx_->session_metadata() if needed.
    return "";
  }

  Status GetConfigProto(const ConfigProto** config_proto) const override;

  // Note: this function is only meant to clear up `config_proto` created by the
  // above `CPluginOpKernelContext::GetConfigProto()`.
  void MaybeDeleteConfigProto(const ConfigProto* config_proto) const override {
    delete config_proto;
  }

  Status GetFunctionLibraryDefinition(
      const FunctionLibraryDefinition** flib_def) const override;

  // Note: this function is only meant to clear up `flib_def` created by the
  // above `CPluginOpKernelContext::GetFunctionLibraryDefinition()`.
  void MaybeDeleteFunctionLibraryDefinition(
      const FunctionLibraryDefinition* flib_def) const override {
    delete flib_def;
  }

  Status GetResourceHandle(int index,
                           const ResourceHandle** handle) const override;

  // Note: this function is only meant to clear up `handle` created by the above
  // `CPluginOpKernelContext::GetResourceHandle()`.
  void MaybeDeleteResourceHandle(const ResourceHandle* handle) const override {
    delete handle;
  }

  int GetGraphDefVersion() const override {
    return TF_GetGraphDefVersion(ctx_);
  }

  Status AllocateOutput(int index, const TensorShape& shape,
                        Tensor** out) override;

  Status SetOutput(int index, const Tensor& tensor) override;

  void CtxFailure(const Status& status) override;
  void CtxFailure(const char* file, int line, const Status& status) override;

  void* GetContext() const override { return ctx_; }

 private:
  mutable mutex mu_;

  // A cache for tensors obtained from the ctx_. This is needed to extend the
  // lifetime of the c++ tensorflow::Tensor created from `TF_TensorToTensor`.
  std::vector<Tensor> obtained_tensors_ TF_GUARDED_BY(mu_);
  TF_OpKernelContext* ctx_;  // not owned.
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_OP_KERNEL_H_
