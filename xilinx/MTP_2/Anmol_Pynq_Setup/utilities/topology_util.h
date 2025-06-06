/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_TOPOLOGY_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_TOPOLOGY_UTIL_H_

#include "span.h"
#include "protocol.pb.h"

namespace xla {

// Given a LocalTopologyProto object from each node, builds a
// GlobalTopologyProto that describes all nodes.
GlobalTopologyProto BuildGlobalTopology(
    absl::Span<LocalTopologyProto> local_topologies);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_TOPOLOGY_UTIL_H_
