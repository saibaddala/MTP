/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_HELPER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_HELPER_H_

#include "kernels.h"
#include "tf_status_helper.h"
#include "c_plugin_coordination_service_agent.h"
#include "direct_plugin_coordination_service_agent.h"
#include "plugin_coordination_service_agent.h"

ABSL_DECLARE_FLAG(bool, next_pluggable_device_use_c_api);

namespace tensorflow {

inline PluginCoordinationServiceAgent* CreatePluginCoordinationServiceAgent(
    void* agent) {
  if (!absl::GetFlag(FLAGS_next_pluggable_device_use_c_api)) {
    return new DirectPluginCoordinationServiceAgent(agent);
  } else {
    return new CPluginCoordinationServiceAgent(agent);
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_HELPER_H_
