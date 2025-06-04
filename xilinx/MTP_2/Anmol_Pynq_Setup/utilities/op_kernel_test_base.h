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

#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_KERNEL_TEST_BASE_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_KERNEL_TEST_BASE_H_

#include <memory>
#include <utility>
#include <vector>

#include "status.h"
#include "str_cat.h"
#include "allocator.h"
#include "attr_value.pb.h"
#include "attr_value_util.h"
#include "fake_input.h"
#include "node_def_builder.h"
#include "op.h"
#include "op_kernel.h"
#include "tensor_shape.pb.h"
#include "tensor_util.h"
#include "types.pb.h"
#include "graph.h"
#include "errors.h"
#include "status_test_util.h"
#include "str_util.h"
#include "strcat.h"
#include "logging.h"
#include "protobuf.h"
#include "test.h"
#include "test_benchmark.h"
#include "error_codes.pb.h"
#include "version.h"
#include "device_name_utils.h"

namespace tensorflow {

static std::vector<DeviceType> DeviceTypes() {
  return {DeviceType(DEVICE_GPU), DeviceType(DEVICE_CPU)};
}

class OpKernelBuilderTest : public ::testing::Test {
 protected:
  // Each attr is described by a "name|type|value".
  NodeDef CreateNodeDef(const string& op_type,
                        const std::vector<string>& attrs) {
    NodeDef node_def;
    node_def.set_name(op_type + "-op");
    node_def.set_op(op_type);
    for (const string& attr_desc : attrs) {
      std::vector<string> parts = str_util::Split(attr_desc, '|');
      CHECK_EQ(parts.size(), 3);
      AttrValue attr_value;
      CHECK(ParseAttrValue(parts[1], parts[2], &attr_value)) << attr_desc;
      node_def.mutable_attr()->insert(
          AttrValueMap::value_type(parts[0], attr_value));
    }
    return node_def;
  }

  std::unique_ptr<OpKernel> ExpectSuccess(const string& op_type,
                                          const DeviceType& device_type,
                                          const std::vector<string>& attrs,
                                          DataTypeSlice input_types = {}) {
    Status status;
    NodeDef def = CreateNodeDef(op_type, attrs);
    for (size_t i = 0; i < input_types.size(); ++i) {
      def.add_input("a:0");
    }

    Env* env = Env::Default();
    DeviceBase device(env);

    // Test CreateOpKernel()
    std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_TRUE(op != nullptr);
    if (op != nullptr) {
      EXPECT_EQ(input_types.size(), op->num_inputs());
      EXPECT_EQ(0, op->num_outputs());
    }

    // Test SupportedDeviceTypesForNode()
    PrioritizedDeviceTypeVector devices;
    TF_EXPECT_OK(SupportedDeviceTypesForNode(DeviceTypes(), def, &devices));
    bool found = false;
    for (const auto& dt : devices) {
      if (dt.first == device_type) {
        found = true;
      }
    }
    EXPECT_TRUE(found) << "Missing " << device_type << " from "
                       << devices.size() << " devices.";

    // In case the caller wants to use the OpKernel
    return op;
  }

  void ExpectFailure(const string& op_type, const DeviceType& device_type,
                     const std::vector<string>& attrs, error::Code code) {
    Status status;
    const NodeDef def = CreateNodeDef(op_type, attrs);
    Env* env = Env::Default();
    DeviceBase device(env);

    // Test CreateOpKernel().
    std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, &device,
                                                cpu_allocator(), def,
                                                TF_GRAPH_DEF_VERSION, &status));
    EXPECT_TRUE(op == nullptr);
    EXPECT_FALSE(status.ok());
    if (!status.ok()) {
      LOG(INFO) << "Status message: " << status.message();
      EXPECT_EQ(code, status.code());

      // Test SupportedDeviceTypesForNode().
      PrioritizedDeviceTypeVector devices;
      if (absl::IsNotFound(status)) {
        TF_EXPECT_OK(SupportedDeviceTypesForNode(DeviceTypes(), def, &devices));
        for (const auto& dt : devices) {
          EXPECT_NE(dt.first, device_type);
        }
      } else {
        Status status2 =
            SupportedDeviceTypesForNode(DeviceTypes(), def, &devices);
        EXPECT_EQ(status.code(), status2.code());
      }
    }
  }

  string GetKernelClassName(const string& op_type,
                            const DeviceType& device_type,
                            const std::vector<string>& attrs,
                            DataTypeSlice input_types = {}) {
    NodeDef def = CreateNodeDef(op_type, attrs);
    for (size_t i = 0; i < input_types.size(); ++i) {
      def.add_input("a:0");
    }

    const KernelDef* kernel_def = nullptr;
    string kernel_class_name;
    const Status status =
        FindKernelDef(device_type, def, &kernel_def, &kernel_class_name);
    if (status.ok()) {
      return kernel_class_name;
    } else if (absl::IsNotFound(status)) {
      return "not found";
    } else {
      return status.ToString();
    }
  }
};

class BaseKernel : public ::tensorflow::OpKernel {
 public:
  explicit BaseKernel(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(::tensorflow::OpKernelContext* context) override {}
  virtual int Which() const = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_KERNEL_TEST_BASE_H_
