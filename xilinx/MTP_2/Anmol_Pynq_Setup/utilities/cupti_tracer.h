/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_TRACER_H_
#define TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_TRACER_H_

#include "optional.h"
#include "cupti.h"
#include "nvToolsExt.h"
#include "cupti_tracer.h"
#include "errors.h"
#include "status.h"
#include "types.h"
#include "cupti_collector.h"
#include "cupti_interface.h"
#include "buffer_pool.h"

namespace tensorflow {
namespace profiler {

using xla::profiler::CuptiTracer;         // NOLINT
using xla::profiler::CuptiTracerOptions;  // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_TRACER_H_
