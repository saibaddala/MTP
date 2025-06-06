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

// This file defines a logger for op names.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_OP_LOGGER_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_OP_LOGGER_H_

#include <memory>
#include <string>

#include "memory.h"
#include "ArrayRef.h"
#include "StringRef.h"
#include "shared_context.h"  // from @tf_runtime
#include "concurrent_vector.h"  // from @tf_runtime
#include "forward_decls.h"  // from @tf_runtime

namespace tfrt {
class HostContext;
}

namespace tensorflow {
namespace tfd {

class OpLogger : public tfrt::SharedContext {
 public:
  explicit OpLogger(tfrt::HostContext* host)
      : op_names_(std::make_unique<tfrt::ConcurrentVector<std::string>>(8)) {}

  void LogOp(tfrt::string_view op_name) {
    op_names_->emplace_back(op_name.str());
  }

  tfrt::ArrayRef<std::string> GetLoggedOps() const {
    absl::Span<const std::string> span = op_names_->ToConstSpan();
    return tfrt::ArrayRef<std::string>(span.data(), span.size());
  }

  // Cannot be called concurrently with any API in this class.
  void Clear() {
    op_names_ = std::make_unique<tfrt::ConcurrentVector<std::string>>(8);
  }

 private:
  std::unique_ptr<tfrt::ConcurrentVector<std::string>> op_names_;
};

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_OP_LOGGER_H_
