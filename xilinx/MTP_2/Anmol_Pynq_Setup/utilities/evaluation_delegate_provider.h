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

#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_DELEGATE_PROVIDER_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_DELEGATE_PROVIDER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "command_line_flags.h"
#include "delegate_provider.h"
#include "evaluation_stages.pb.h"
#include "utils.h"
#include "tool_params.h"

namespace tflite {
namespace evaluation {

using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;
class DelegateProviders {
 public:
  DelegateProviders();

  // Returns a list of commandline flags that delegate providers define.
  std::vector<Flag> GetFlags();

  // Initialize delegate-related parameters from commandline arguments and
  // returns true if successful.
  bool InitFromCmdlineArgs(int* argc, const char** argv);

  // Get all parameters from all registered delegate providers.
  const tools::ToolParams& GetAllParams() const { return params_; }

  // Get a new set of parameters based on the given TfliteInferenceParams
  // 'params' but considering what have been initialized (i.e. 'params_').
  // Note the same-meaning parameter (e.g. number of TfLite interpreter threads)
  // in TfliteInferenceParams will take precedence over the parameter of the
  // same meaning in 'params_'.
  tools::ToolParams GetAllParams(const TfliteInferenceParams& params) const;

  // Create the a TfLite delegate instance based on the provided delegate
  // 'name'. If the specified one isn't found, an empty TfLiteDelegatePtr is
  // returned.
  TfLiteDelegatePtr CreateDelegate(const std::string& name) const;

  // Create a list of TfLite delegates based on what have been initialized (i.e.
  // 'params_').
  std::vector<ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates()
      const {
    return delegate_list_util_.CreateAllRankedDelegates();
  }

  // Create a list of TfLite delegates based on the given TfliteInferenceParams
  // 'params' but considering what have been initialized (i.e. 'params_').
  std::vector<ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates(
      const TfliteInferenceParams& params) const {
    auto converted = GetAllParams(params);
    ProvidedDelegateList util(&converted);
    return util.CreateAllRankedDelegates();
  }

 private:
  // Contain delegate-related parameters that are initialized from command-line
  // flags.
  tools::ToolParams params_;

  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;

  // Key is the delegate name, and the value is the index to a TfLite delegate
  // provider in the "delegate_list_util_.providers()" list.
  const std::unordered_map<std::string, int> delegates_map_;
};

// Parse a string 'val' to the corresponding delegate type defined by
// TfliteInferenceParams::Delegate.
TfliteInferenceParams::Delegate ParseStringToDelegateType(
    const std::string& val);

// Create a TfLite delegate based on the given TfliteInferenceParams 'params'.
// If there's an error during the creation, an error message will be recorded to
// 'error_msg' if provided.
TfLiteDelegatePtr CreateTfLiteDelegate(const TfliteInferenceParams& params,
                                       std::string* error_msg = nullptr);
}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_DELEGATE_PROVIDER_H_
