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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_OP_KERNEL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_OP_KERNEL_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pjrt_client.h"
#include "attr_value.pb.h"
#include "types.h"
#include "status.h"

namespace tensorflow {

class ConfigProto;
class FunctionLibraryDefinition;
class OpInputList;
class PluginCoordinationServiceAgent;
class PluginVariable;
class Tensor;
class TensorShape;

// A wrapper base class that provides convenience for developers to implement
// to plugin OpKernels that suites internal and external requirements, without
// duplicating code.
//
// Internal build: Plugin and TF are built together and statically linked. In
// this case, we can directly cast between `TF_OpKernelContext*` and
// `OpKernelContext*`, and directly call C++ API. This way don't need to pay the
// potential performance panelty (e.g. proto serialization/deserialization)
// brought by C API.
//
// External build: Plugin and TF are built separately (potentially on different
// platform and by different compilers). Plugin is dynamically loaded by TF.
// In this case, we need to call C API to ensure binary compatibility.
//
// `DirectPluginOpKernel*` and `CPluginOpKernel*` implement `PluginOpKernel*`
// to support the above mentioned internal and external build cases. OpKernel
// developers can conveniently use the `Wrapper` C++ API to implement `Create`
// and `Compute` functions, and use the helper macro to register the functions
// as a Plugin OpKernel. This method benefit kernel developers in two ways: 1).
// Plugin OpKernel developers don't have to directly deal with C API. 2). In the
// OpKernels are performance critical and developers want to introduce an
// internal version of the same OpKernels, they don't have to implement again
// with mostly duplicated code.
class PluginOpKernelConstruction {
 public:
  PluginOpKernelConstruction() = default;
  virtual ~PluginOpKernelConstruction() = default;

  virtual Status GetBoolAttr(std::string_view attr_name, bool* value) const = 0;
  virtual Status GetInt32Attr(std::string_view attr_name, int* value) const = 0;
  virtual Status GetInt32AttrList(std::string_view attr_name,
                                  std::vector<int32_t>* value) const = 0;
  virtual Status GetInt64Attr(std::string_view attr_name,
                              int64_t* value) const = 0;
  virtual Status GetStringAttr(std::string_view attr_name,
                               std::string* value) const = 0;
  virtual Status GetFunctionAttr(std::string_view attr_name,
                                 NameAttrList* function) const = 0;

  virtual void CtxFailure(const Status& status) = 0;
  virtual void CtxFailure(const char* file, int line, const Status& status) = 0;

  virtual void* GetContext() const = 0;
};

class PluginOpKernelContext {
 public:
  PluginOpKernelContext() = default;
  virtual ~PluginOpKernelContext() = default;

  virtual std::string_view GetResourceMgrDefaultContainerName() = 0;

  virtual Status LookupOrCreateResource(std::string_view container_name,
                                        std::string_view plugin_resource_name,
                                        void** result_plugin_resource,
                                        void* (*create_func)(void*),
                                        void* create_func_args,
                                        void (*delete_func)(void*)) = 0;

  virtual PluginCoordinationServiceAgent* GetPluginCoordinationServiceAgent()
      const = 0;

  // This method will allocate a new `PluginVariable`. Caller is responsible
  // for managing it's lifetime.
  virtual Status CreatePluginVariable(int index,
                                      PluginVariable** variable) const = 0;

  virtual Status AllocateTempForPluginVariable(PluginVariable* variable) = 0;

  virtual int NumInputs() const = 0;

  virtual Status GetInput(int index, Tensor* tensor) const = 0;

  // This method is not marked const because CPluginOpKernel need to do some
  // extra bookkeeping work.
  virtual Status GetInput(const char* name, const Tensor** tensor) = 0;

  virtual Status GetInputRange(std::string_view name,
                               std::pair<int, int>* range) const = 0;

  virtual DataType GetInputDataType(int index) const = 0;

  virtual std::string_view GetOpKernelRequestedInput(int index) const = 0;

  virtual std::string_view GetOpKernelName() const = 0;

  virtual uint64_t GetFrameId() const = 0;

  virtual int64_t GetIterId() const = 0;

  virtual int64_t GetStepId() const = 0;

  virtual int GetDeviceId() const = 0;

  virtual std::string GetSessionName() const = 0;

  virtual Status GetConfigProto(const ConfigProto** config_proto) const = 0;

  virtual void MaybeDeleteConfigProto(
      const ConfigProto* config_proto) const = 0;

  virtual Status GetFunctionLibraryDefinition(
      const FunctionLibraryDefinition** flib_def) const = 0;

  virtual void MaybeDeleteFunctionLibraryDefinition(
      const FunctionLibraryDefinition* flib_def) const = 0;

  virtual Status GetResourceHandle(int index,
                                   const ResourceHandle** handle) const = 0;

  virtual void MaybeDeleteResourceHandle(
      const ResourceHandle* handle) const = 0;

  virtual int GetGraphDefVersion() const = 0;

  virtual Status AllocateOutput(int index, const TensorShape& shape,
                                Tensor** out) = 0;

  virtual Status SetOutput(int index, const Tensor& tensor) = 0;

  virtual void CtxFailure(const Status& status) = 0;
  virtual void CtxFailure(const char* file, int line, const Status& status) = 0;

  virtual void* GetContext() const = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_OP_KERNEL_H_
