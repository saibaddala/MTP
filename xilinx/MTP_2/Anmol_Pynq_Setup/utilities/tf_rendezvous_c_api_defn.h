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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_RENDEZVOUS_C_API_DEFN_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_RENDEZVOUS_C_API_DEFN_H_

#include "tf_rendezvous_c_api.h"
#include "cancellation.h"
#include "device_base.h"
#include "tensor.h"

struct TF_DeviceContext {
  tensorflow::DeviceContext* device_context;  // not owned
};

struct TF_CancellationManager {
  tensorflow::CancellationManager* cancellation_manager;  // not owned
};

struct TF_TensorWrapper {
  tensorflow::Tensor tensor;
};

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_RENDEZVOUS_C_API_DEFN_H_
