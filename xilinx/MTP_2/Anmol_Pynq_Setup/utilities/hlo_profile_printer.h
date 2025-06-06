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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PROFILE_PRINTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PROFILE_PRINTER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "hlo_profile_printer_data.pb.h"
#include "types.h"

namespace xla {
// Pretty-print an array of profile counters using hlo_profile_printer_data.
std::string PrintHloProfile(
    const HloProfilePrinterData& hlo_profile_printer_data,
    const int64_t* counters, double clock_rate_ghz);
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PROFILE_PRINTER_H_
