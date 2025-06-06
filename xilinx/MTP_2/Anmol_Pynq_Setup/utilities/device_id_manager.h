/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_FRAMEWORK_DEVICE_ID_MANAGER_H_
#define TENSORFLOW_TSL_FRAMEWORK_DEVICE_ID_MANAGER_H_

#include "device_id.h"
#include "device_type.h"
#include "status.h"

namespace tsl {

// Class that maintains a map from TfDeviceId to PlatformDeviceId, and manages
// the translation between them.
class DeviceIdManager {
 public:
  // Adds a mapping from tf_device_id to platform_device_id.
  static Status InsertTfPlatformDeviceIdPair(
      const DeviceType& type, TfDeviceId tf_device_id,
      PlatformDeviceId platform_device_id);

  // Gets the platform_device_id associated with tf_device_id. Returns OK if
  // found.
  static Status TfToPlatformDeviceId(const DeviceType& type,
                                     TfDeviceId tf_device_id,
                                     PlatformDeviceId* platform_device_id);

  // Clears the map. Used in unit tests only.
  static void TestOnlyReset();
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_FRAMEWORK_DEVICE_ID_MANAGER_H_
