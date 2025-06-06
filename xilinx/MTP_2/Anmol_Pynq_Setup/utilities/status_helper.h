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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_STATUS_HELPER_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_STATUS_HELPER_H_

#include "tpu_api.h"
#include "tpu_executor_c_api.h"
#include "status.h"
#include "error_codes.pb.h"

class StatusHelper {
 public:
  StatusHelper()
      : c_status(stream_executor::tpu::ExecutorApiFn()->TpuStatus_NewFn()) {}

  ~StatusHelper() {
    stream_executor::tpu::ExecutorApiFn()->TpuStatus_FreeFn(c_status);
  }

  static tsl::Status FromC(  // TENSORFLOW_STATUS_OK
      TF_Status* const c_status) {
    if (stream_executor::tpu::ExecutorApiFn()->TpuStatus_OkFn(c_status)) {
      return ::tsl::OkStatus();
    } else {
      return tsl::Status(  // TENSORFLOW_STATUS_OK
          absl::StatusCode(
              stream_executor::tpu::ExecutorApiFn()->TpuStatus_CodeFn(
                  c_status)),
          stream_executor::tpu::ExecutorApiFn()->TpuStatus_MessageFn(c_status));
    }
  }

  bool ok() const {
    return stream_executor::tpu::ExecutorApiFn()->TpuStatus_OkFn(c_status);
  }

  tsl::Status status() const {  // TENSORFLOW_STATUS_OK
    return FromC(c_status);
  }

  TF_Status* const c_status;  // NOLINT
};

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_STATUS_HELPER_H_
