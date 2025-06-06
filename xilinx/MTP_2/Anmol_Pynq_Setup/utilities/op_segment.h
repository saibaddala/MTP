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

#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_SEGMENT_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_SEGMENT_H_

#include <string>
#include <unordered_map>

#include "op_kernel.h"
#include "status.h"
#include "macros.h"
#include "mutex.h"
#include "thread_annotations.h"
#include "types.h"

namespace tensorflow {

// OpSegment keeps track of OpKernels registered for sessions running
// on a device.
//
// The implementation maintains a two-level map. The 1st level maps
// session handle to the map of registered OpKernels. The 2nd level
// map maps node names to instantiated OpKernel objects.
//
// Each 2-nd level map is reference-counted and the caller can call
// AddHold to obtain a reference on all kernels of a session and
// ensure these kernels are alive until a corresponding RemoveHold is
// called on the same session.
class OpSegment {
 public:
  OpSegment();
  ~OpSegment();

  // A hold can be placed on a session, preventing all its kernels
  // from being deleted.
  void AddHold(const std::string& session_handle);
  void RemoveHold(const std::string& session_handle);

  // If the kernel for "node_name" has been created in the
  // "session_handle", returns the existing op kernel in "*kernel".
  // Otherwise, creates the kernel by calling create_fn(), cache it,
  // and returns it in "*kernel". If create_fn() fails, returns the
  // error.
  //
  // OpSegment keeps the ownership of the returned "*kernel".
  typedef std::function<Status(OpKernel**)> CreateKernelFn;
  Status FindOrCreate(const std::string& session_handle,
                      const std::string& node_name, OpKernel** kernel,
                      CreateKernelFn create_fn);

  // Returns true if OpSegment should own the kernel.
  static bool ShouldOwnKernel(FunctionLibraryRuntime* lib,
                              const std::string& node_op);

 private:
  // op name -> OpKernel
  typedef std::unordered_map<string, OpKernel*> KernelMap;
  struct Item {
    int num_holds = 1;      // Num of holds put on the session.
    KernelMap name_kernel;  // op name -> kernel.
    ~Item();
  };

  // session handle -> item.
  // Session handles are produced by strings::FpToString()
  typedef std::unordered_map<string, Item*> SessionMap;

  mutable mutex mu_;
  SessionMap sessions_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(OpSegment);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_SEGMENT_H_
