/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_CONV_2D_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_CONV_2D_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest.h>
#include "common.h"
#include "xnnpack_delegate.h"
#include "schema_generated.h"

namespace tflite {
namespace xnnpack {

class Conv2DTester {
 public:
  enum class WeightsType {
    kFP32,
    kFP16,
    kTensorWiseQuantizedInt8,
    kChannelWiseQuantizedInt8,
  };
  enum class BiasType {
    kFP32,
    kFP16,
  };

  Conv2DTester() = default;
  Conv2DTester(const Conv2DTester&) = delete;
  Conv2DTester& operator=(const Conv2DTester&) = delete;

  inline Conv2DTester& BatchSize(int32_t batch_size) {
    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const { return batch_size_; }

  inline Conv2DTester& InputChannels(int32_t input_channels) {
    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const { return input_channels_; }

  inline Conv2DTester& OutputChannels(int32_t output_channels) {
    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const { return output_channels_; }

  inline Conv2DTester& Groups(int32_t groups) {
    EXPECT_EQ(InputChannels() % groups, 0);
    EXPECT_EQ(OutputChannels() % groups, 0);
    groups_ = groups;
    return *this;
  }

  inline int32_t Groups() const { return groups_; }

  inline int32_t KernelInputChannels() const {
    return input_channels_ / groups_;
  }

  inline Conv2DTester& InputHeight(int32_t input_height) {
    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  inline int32_t InputHeight() const { return input_height_; }

  inline Conv2DTester& InputWidth(int32_t input_width) {
    EXPECT_GT(input_width, 0);
    input_width_ = input_width;
    return *this;
  }

  inline int32_t InputWidth() const { return input_width_; }

  inline int32_t OutputWidth() const {
    if (Padding() == ::tflite::Padding_SAME) {
      EXPECT_GE(InputWidth(), 1);
      return (InputWidth() - 1) / StrideWidth() + 1;
    } else {
      EXPECT_GE(InputWidth(), DilatedKernelWidth());
      return 1 + (InputWidth() - DilatedKernelWidth()) / StrideWidth();
    }
  }

  inline int32_t OutputHeight() const {
    if (Padding() == ::tflite::Padding_SAME) {
      EXPECT_GE(InputHeight(), 1);
      return (InputHeight() - 1) / StrideHeight() + 1;
    } else {
      EXPECT_GE(InputHeight(), DilatedKernelHeight());
      return 1 + (InputHeight() - DilatedKernelHeight()) / StrideHeight();
    }
  }

  inline Conv2DTester& KernelHeight(int32_t kernel_height) {
    EXPECT_GT(kernel_height, 0);
    kernel_height_ = kernel_height;
    return *this;
  }

  inline int32_t KernelHeight() const { return kernel_height_; }

  inline Conv2DTester& KernelWidth(int32_t kernel_width) {
    EXPECT_GT(kernel_width, 0);
    kernel_width_ = kernel_width;
    return *this;
  }

  inline int32_t KernelWidth() const { return kernel_width_; }

  inline Conv2DTester& StrideHeight(int32_t stride_height) {
    EXPECT_GT(stride_height, 0);
    stride_height_ = stride_height;
    return *this;
  }

  inline int32_t StrideHeight() const { return stride_height_; }

  inline Conv2DTester& StrideWidth(int32_t stride_width) {
    EXPECT_GT(stride_width, 0);
    stride_width_ = stride_width;
    return *this;
  }

  inline int32_t StrideWidth() const { return stride_width_; }

  inline Conv2DTester& DilationHeight(int32_t dilation_height) {
    EXPECT_GT(dilation_height, 0);
    dilation_height_ = dilation_height;
    return *this;
  }

  inline int32_t DilationHeight() const { return dilation_height_; }

  inline Conv2DTester& DilationWidth(int32_t dilation_width) {
    EXPECT_GT(dilation_width, 0);
    dilation_width_ = dilation_width;
    return *this;
  }

  inline int32_t DilationWidth() const { return dilation_width_; }

  inline int32_t DilatedKernelHeight() const {
    return (KernelHeight() - 1) * DilationHeight() + 1;
  }

  inline int32_t DilatedKernelWidth() const {
    return (KernelWidth() - 1) * DilationWidth() + 1;
  }

  inline Conv2DTester& FP16Weights() {
    weights_type_ = WeightsType::kFP16;
    bias_type_ = BiasType::kFP16;
    return *this;
  }

  inline Conv2DTester& TensorWiseQuantizedInt8Weights() {
    weights_type_ = WeightsType::kTensorWiseQuantizedInt8;
    // Bias is stored in FP32 even when filter is quantized to INT8
    bias_type_ = BiasType::kFP32;
    return *this;
  }

  inline Conv2DTester& ChannelWiseQuantizedInt8Weights() {
    weights_type_ = WeightsType::kChannelWiseQuantizedInt8;
    // Bias is stored in FP32 even when filter is quantized to INT8
    bias_type_ = BiasType::kFP32;
    return *this;
  }

  inline Conv2DTester& SparseWeights() {
    sparse_weights_ = true;
    return *this;
  }

  inline bool SparseWeights() const { return sparse_weights_; }

  inline Conv2DTester& SamePadding() {
    padding_ = ::tflite::Padding_SAME;
    return *this;
  }

  inline Conv2DTester& ValidPadding() {
    padding_ = ::tflite::Padding_VALID;
    return *this;
  }

  inline Conv2DTester& ReluActivation() {
    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline Conv2DTester& Relu6Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline Conv2DTester& ReluMinus1To1Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  inline Conv2DTester& TanhActivation() {
    activation_ = ::tflite::ActivationFunctionType_TANH;
    return *this;
  }

  inline Conv2DTester& SignBitActivation() {
    activation_ = ::tflite::ActivationFunctionType_SIGN_BIT;
    return *this;
  }

  inline Conv2DTester& WeightsCache(
      TfLiteXNNPackDelegateWeightsCache* weights_cache) {
    weights_cache_ = weights_cache;
    return *this;
  }

  void Test(TfLiteDelegate* delegate) const;

  std::vector<char> CreateTfLiteModel() const;

 private:
  inline WeightsType WeightsType() const { return weights_type_; }

  inline BiasType BiasType() const { return bias_type_; }

  inline ::tflite::Padding Padding() const { return padding_; }

  inline ::tflite::ActivationFunctionType Activation() const {
    return activation_;
  }

  int32_t batch_size_ = 1;
  int32_t input_channels_ = 1;
  int32_t output_channels_ = 1;
  int32_t groups_ = 1;
  int32_t input_height_ = 1;
  int32_t input_width_ = 1;
  int32_t kernel_height_ = 1;
  int32_t kernel_width_ = 1;
  int32_t stride_height_ = 1;
  int32_t stride_width_ = 1;
  int32_t dilation_height_ = 1;
  int32_t dilation_width_ = 1;
  enum WeightsType weights_type_ { WeightsType::kFP32 };
  enum BiasType bias_type_ { BiasType::kFP32 };
  bool sparse_weights_ = false;
  ::tflite::Padding padding_ = ::tflite::Padding_VALID;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
  TfLiteXNNPackDelegateWeightsCache* weights_cache_ = nullptr;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_CONV_2D_TESTER_H_
