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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_WORKER_CLIENT_H_
#define TENSORFLOW_CORE_DATA_SERVICE_WORKER_CLIENT_H_

#include <memory>
#include <string>

#include "common.h"
#include "common.pb.h"
#include "data_transfer.h"
#include "worker.pb.h"
#include "mutex.h"
#include "status.h"
#include "statusor.h"
#include "types.h"

namespace tensorflow {
namespace data {

constexpr const char kLocalTransferProtocol[] = "local";
constexpr const char kGrpcTransferProtocol[] = "grpc";

// Client for communicating with the tf.data service worker.
class DataServiceWorkerClient : public DataServiceClientBase {
 public:
  DataServiceWorkerClient(const std::string& address,
                          const std::string& protocol,
                          const std::string& transfer_protocol)
      : DataServiceClientBase(address, protocol),
        transfer_protocol_(transfer_protocol) {}

  // Fetches an element from the worker.
  Status GetElement(const GetElementRequest& req, GetElementResult& result);

  // Makes a best effort to cancel all outstanding calls in progress for the
  // client, and causes further calls to return Cancelled status.
  void TryCancel();
  // Returns an error if the client is incompatible with a server which has the
  // properties described in `compatibility_info`.
  Status CheckCompatibility(
      const std::string& server_compatibility_info) const {
    return client_->CheckCompatibility(server_compatibility_info);
  }
  // Returns the data transfer protocol, preferring to use the local transfer
  // protocol if a local tf.data worker exists.
  std::string GetDataTransferProtocol() const;

 protected:
  Status EnsureInitialized() override;

 private:
  const std::string transfer_protocol_;
  mutex mu_;
  // Initialization is guarded by `mu_`, but using the stub does not require
  // holding `mu_`
  std::unique_ptr<DataTransferClient> client_;
};

// Creates and initializes a new tf.data service worker client to read
// from the data transfer server specified in `info`.
StatusOr<std::unique_ptr<DataServiceWorkerClient>>
CreateDataServiceWorkerClient(const std::string& dispatcher_protocol,
                              const DataTransferServerInfo& info);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_WORKER_CLIENT_H_
