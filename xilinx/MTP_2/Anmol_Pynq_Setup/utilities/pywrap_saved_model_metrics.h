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

// Defines the 'metrics' wrapper submodule.

#ifndef TENSORFLOW_PYTHON_SAVED_MODEL_PYWRAP_SAVED_MODEL_METRICS_H_
#define TENSORFLOW_PYTHON_SAVED_MODEL_PYWRAP_SAVED_MODEL_METRICS_H_

#include "pybind11.h"  // from @pybind11

namespace tensorflow {
namespace saved_model {
namespace python {

// Wraps the SM and Checkpoint Metrics API methods for exporting to Python.
void DefineMetricsModule(pybind11::module main_module);

}  // namespace python
}  // namespace saved_model
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_SAVED_MODEL_PYWRAP_SAVED_MODEL_METRICS_H_
