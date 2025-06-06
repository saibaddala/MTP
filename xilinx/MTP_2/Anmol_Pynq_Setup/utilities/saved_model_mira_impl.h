/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_MIRA_IMPL_H_
#define TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_MIRA_IMPL_H_

// This file contains stub implementations for Google internal
// `SavedModelMiraImpl` APIs.

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "saved_model.h"

namespace tensorflow {
namespace tfrt_stub {

class SavedModelMiraImpl final : public SavedModel {
 public:
  tensorflow::StatusOr<std::unique_ptr<SavedModel>> LoadSavedModel(
      Options options, absl::string_view saved_model_dir,
      const std::unordered_set<std::string>& tags);
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_MIRA_IMPL_H_
