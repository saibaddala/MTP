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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_TASKS_TASK_EXECUTOR_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_TASKS_TASK_EXECUTOR_H_

#include <optional>

#include "optional.h"
#include "command_line_flags.h"
#include "evaluation_delegate_provider.h"
#include "evaluation_config.pb.h"

namespace tflite {
namespace evaluation {
// A common task execution API to avoid boilerpolate code in defining the main
// function.
class TaskExecutor {
 public:
  virtual ~TaskExecutor() {}

  // If the run is successful, the latest metrics will be returned.
  std::optional<EvaluationStageMetrics> Run(int* argc, char* argv[]);

 protected:
  // Returns a list of commandline flags that this task defines.
  virtual std::vector<Flag> GetFlags() = 0;

  virtual std::optional<EvaluationStageMetrics> RunImpl() = 0;

  DelegateProviders delegate_providers_;
};

// Just a declaration. In order to avoid the boilerpolate main-function code,
// every evaluation task should define this function.
std::unique_ptr<TaskExecutor> CreateTaskExecutor();
}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_TASKS_TASK_EXECUTOR_H_
