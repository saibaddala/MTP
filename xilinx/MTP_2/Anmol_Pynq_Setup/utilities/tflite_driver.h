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
#ifndef TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_
#define TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_

#include <stdlib.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "op_resolver.h"
#include "model_builder.h"
#include "interpreter.h"
#include "signature_runner.h"
#include "result_expectations.h"
#include "test_runner.h"

namespace tflite {
namespace testing {

// A test runner that feeds inputs into TF Lite and verifies its outputs.
class TfLiteDriver : public TestRunner {
 public:
  enum class DelegateType {
    kNone,
    kNnapi,
    kGpu,
    kFlex,
  };

  // Initialize the global test delegate providers from commandline arguments
  // and returns true if successful.
  static bool InitTestDelegateProviders(int* argc, const char** argv);

  /**
   * Creates a new TfLiteDriver
   * @param  delegate         The (optional) delegate to use.
   * @param  reference_kernel Whether to use the builtin reference kernel
   * ops.
   */
  explicit TfLiteDriver(DelegateType delegate_type = DelegateType::kNone,
                        bool reference_kernel = false);
  ~TfLiteDriver() override;

  void LoadModel(const std::string& bin_file_path) override;
  void LoadModel(const std::string& bin_file_path,
                 const std::string& signature) override;

  void ReshapeTensor(const std::string& name,
                     const std::string& csv_values) override;
  void ResetTensor(const std::string& name) override;
  std::string ReadOutput(const std::string& name) override;
  void Invoke(
      const std::vector<std::pair<std::string, std::string>>& inputs) override;
  bool CheckResults(
      const std::vector<std::pair<std::string, std::string>>& expected_outputs,
      const std::vector<std::pair<std::string, std::string>>&
          expected_output_shapes) override;
  std::vector<std::string> GetOutputNames() override;

  void AllocateTensors() override;
  void SetThreshold(double relative_threshold, double absolute_threshold);
  void SetQuantizationErrorMultiplier(int quantization_error_multiplier);

 protected:
  Interpreter::TfLiteDelegatePtr delegate_;

 private:
  void SetInput(const std::string& name, const std::string& csv_values);
  void SetExpectation(const std::string& name, const std::string& csv_values);
  void SetShapeExpectation(const std::string& name,
                           const std::string& csv_values);
  void DeallocateStringTensor(TfLiteTensor* t) {
    if (t) {
      free(t->data.raw);
      t->data.raw = nullptr;
    }
  }
  void AllocateStringTensor(int id, size_t num_bytes, TfLiteTensor* t) {
    t->data.raw = reinterpret_cast<char*>(malloc(num_bytes));
    t->bytes = num_bytes;
    tensors_to_deallocate_[id] = t;
  }

  void ResetLSTMStateTensors();
  // Formats tensor value to string in csv format.
  std::string TensorValueToCsvString(const TfLiteTensor* tensor);

  std::map<std::string, uint32_t> signature_inputs_;
  std::map<std::string, uint32_t> signature_outputs_;
  std::unique_ptr<OpResolver> resolver_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
  std::map<int, std::unique_ptr<DataExpectation>> expected_output_;
  std::map<int, std::unique_ptr<ShapeExpectation>> expected_output_shape_;
  SignatureRunner* signature_runner_ = nullptr;
  bool must_allocate_tensors_ = true;
  std::map<int, TfLiteTensor*> tensors_to_deallocate_;
  double relative_threshold_;
  double absolute_threshold_;
  int quantization_error_multiplier_;
};

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_
