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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_TRANSFER_MANAGER_H_

#include <vector>

#include "span.h"
#include "literal.h"
#include "xfeed_manager.h"
#include "generic_transfer_manager.h"
#include "transfer_manager.h"
#include "statusor.h"
#include "device_memory.h"
#include "stream_executor.h"
#include "xla_data.pb.h"

namespace xla {

// An implementation of the XLA GenericTransferManager that
// handles CPU-specific infeed.
class CpuTransferManager : public GenericTransferManager {
 public:
  CpuTransferManager();
  ~CpuTransferManager() override {}

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal) override;
  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    MutableBorrowingLiteral literal) override;

  bool CanShapedBufferBeAccessedNow(
      se::StreamExecutor* executor,
      const ShapedBuffer& device_buffer) const override {
    return true;
  }

  bool CanBufferBeAccessedNow(
      se::StreamExecutor* executor,
      const se::DeviceMemoryBase& device_buffer) const override {
    return true;
  }

  Status ReadDynamicShapes(se::Stream* stream,
                           const ShapedBuffer* device_buffer,
                           Shape* device_shape) override;

 private:
  CpuTransferManager(const CpuTransferManager&) = delete;
  CpuTransferManager& operator=(const CpuTransferManager&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_TRANSFER_MANAGER_H_
