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
#ifndef TENSORFLOW_LITE_C_C_API_OPAQUE_INTERNAL_H_
#define TENSORFLOW_LITE_C_C_API_OPAQUE_INTERNAL_H_

#include <memory>

#include "op_resolver.h"
#include "common.h"

// Internal structures and subroutines used by the C API. These are likely to
// change and should not be depended on directly by any C API clients.
//
// NOTE: This header does not follow C conventions and does not define a C API.
// It is effectively an (internal) implementation detail of the C API.

namespace tflite {
namespace internal {

class CommonOpaqueConversionUtil {
 public:
  // Obtain (or create) a 'TfLiteRegistrationExternal' object that corresponds
  // to the provided 'registration' argument, and return the address of the
  // external registration.  We loosely define that a
  // 'TfLiteRegistrationExternal' object "corresponds" to a 'TfLiteRegistration'
  // object when calling any function pointer (like 'prepare') on the
  // 'TfLiteRegistrationExternal' object calls into the corresponding function
  // pointer of the 'TfLiteRegistration' object.
  //
  // The specified 'context' or 'op_resolver' object is used to store the
  // 'TfLiteRegistrationExternal*' pointers. The 'TfLiteRegistrationExternal*'
  // pointer will be deallocated when that object gets destroyed.  I.e., the
  // caller of this function should not deallocate the object pointed to by the
  // return value of 'ObtainRegistrationExternal'.
  //
  // We also need to provide the 'node_index' that the 'registration'
  // corresponds to, so that the 'TfLiteRegistrationExternal' can store that
  // index within its fields.  If the registration does not yet correspond
  // to a specific node index, then 'node_index' should be -1.
  static TfLiteRegistrationExternal* ObtainRegistrationExternal(
      TfLiteContext* context, const TfLiteRegistration* registration,
      int node_index);

  // Get a shared_ptr to the RegistrationExternalsCache from an OpResolver.
  // This is used to allow the InterpreterBuilder and OpResolver to share
  // the same RegistrationExternalsCache, so that the RegistrationExternal
  // objects in it can persist for the lifetimes of both the InterpreterBuilder
  // and OpResolver.
  static std::shared_ptr<::tflite::internal::RegistrationExternalsCache>
  GetSharedCache(const ::tflite::OpResolver& op_resolver) {
    return op_resolver.registration_externals_cache_;
  }

 private:
  static TfLiteRegistrationExternal* CachedObtainRegistrationExternal(
      ::tflite::internal::RegistrationExternalsCache*
          registration_externals_cache,
      const TfLiteRegistration* registration, int node_index);
};

}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_C_C_API_OPAQUE_INTERNAL_H_
