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

#ifndef TENSORFLOW_COMPILER_MLIR_PYTHON_MLIR_WRAPPER_MLIR_WRAPPER_H_
#define TENSORFLOW_COMPILER_MLIR_PYTHON_MLIR_WRAPPER_MLIR_WRAPPER_H_

#include "pybind11.h"  // from @pybind11
#include "stl.h"  // from @pybind11

namespace py = pybind11;

void init_basic_classes(py::module& m);
void init_types(py::module& m);
void init_builders(py::module& m);
void init_ops(py::module& m);
void init_attrs(py::module& m);

#endif  // TENSORFLOW_COMPILER_MLIR_PYTHON_MLIR_WRAPPER_MLIR_WRAPPER_H_
