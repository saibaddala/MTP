/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_
#define TENSORFLOW_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_

#include <memory>
#include <string>
#include <vector>

#include "device.h"
#include "optimization_registry.h"
#include "status.h"
#include "config.pb.h"

namespace tensorflow {

// OptimizationPassRunner can be initialized, populated with devices, then run
// to test individual Tensorflow Optimization passes.
class OptimizationPassRunner {
 public:
  explicit OptimizationPassRunner() : jit_level_(OptimizerOptions::DEFAULT) {}

  // Increasing the Jit level will cause XLA to compile parts of the tensorflow
  // graph that it is able to.
  Status SetJitLevel(OptimizerOptions::GlobalJitLevel jit_level);

  Status Run(absl::string_view pass_to_run, GraphDef input, GraphDef* result);

  Status AddCpus(int count) {
    return AddDevices(tensorflow::DEVICE_CPU, count);
  }

  Status AddGpus(int count) {
    return AddDevices(tensorflow::DEVICE_GPU, count);
  }

 private:
  Status AddDevices(absl::string_view type, int count);

  OptimizerOptions::GlobalJitLevel jit_level_;
  std::vector<std::unique_ptr<Device>> devices_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_
