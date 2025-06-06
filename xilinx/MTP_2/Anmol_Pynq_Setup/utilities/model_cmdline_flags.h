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
#ifndef TENSORFLOW_LITE_TOCO_MODEL_CMDLINE_FLAGS_H_
#define TENSORFLOW_LITE_TOCO_MODEL_CMDLINE_FLAGS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "model_flags.pb.h"
#include "types.pb.h"

namespace toco {
// Parse and remove arguments for models (in toco). Returns true if parsing
// is successful. msg has the usage string if there was an error or
// "--help" was specified
bool ParseModelFlagsFromCommandLineFlags(
    int* argc, char* argv[], std::string* msg,
    ParsedModelFlags* parsed_model_flags_ptr);
// Populate the ModelFlags proto with model data.
void ReadModelFlagsFromCommandLineFlags(
    const ParsedModelFlags& parsed_model_flags, ModelFlags* model_flags);
// Parse the global model flags to a static
void ParseModelFlagsOrDie(int* argc, char* argv[]);
// Get the global parsed model flags
ParsedModelFlags* GlobalParsedModelFlags();

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_MODEL_CMDLINE_FLAGS_H_
