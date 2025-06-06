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

#ifndef TENSORFLOW_CORE_TFRT_SAVED_MODEL_UTILS_SERIALIZE_BEF_UTILS_H_
#define TENSORFLOW_CORE_TFRT_SAVED_MODEL_UTILS_SERIALIZE_BEF_UTILS_H_

#include <memory>
#include <string>

#include "status.h"
#include "statusor.h"
#include "ToolOutputFile.h"
#include "FileUtilities.h"  // from @llvm-project
#include "dump_mlir_util.h"
#include "status.h"
#include "env.h"
#include "bef_buffer.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// Serializes the BefBuffer into a file
absl::Status SerializeBEF(const tfrt::BefBuffer &bef,
                          const std::string &filepath);

// Deserializes BEF file from filepath into a BEFBuffer
absl::StatusOr<tfrt::BefBuffer> DeserializeBEFBuffer(
    const std::string &filepath);
}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_UTILS_SERIALIZE_BEF_UTILS_H_
