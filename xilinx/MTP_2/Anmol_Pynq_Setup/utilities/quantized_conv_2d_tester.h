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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_CONV_2D_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_CONV_2D_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest.h>
#include "common.h"
#include "xnnpack_delegate.h"
#include "interpreter.h"
#include "schema_generated.h"

namespace tflite {
namespace xnnpack {

// Creates a model with a single CONV_2D operator with quantized input, output,
// and weights, runs this model in two TensorFlow Lite interpreters, one with
// the delegate applied, and the other without, and compares the results.
class QuantizedConv2DTester {
 public:
  QuantizedConv2DTester() = default;
  QuantizedConv2DTester(const QuantizedConv2DTester&) = delete;
  QuantizedConv2DTester& operator=(const QuantizedConv2DTester&) = delete;

  inline QuantizedConv2DTester& BatchSize(int32_t batch_size) {
    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  inline int32_t BatchSize() const { return batch_size_; }

  inline QuantizedConv2DTester& InputChannels(int32_t input_channels) {
    EXPECT_GT(input_channels, 0);
    input_channels_ = input_channels;
    return *this;
  }

  inline int32_t InputChannels() const { return input_channels_; }

  inline QuantizedConv2DTester& OutputChannels(int32_t output_channels) {
    EXPECT_GT(output_channels, 0);
    output_channels_ = output_channels;
    return *this;
  }

  inline int32_t OutputChannels() const { return output_channels_; }

  inline QuantizedConv2DTester& InputHeight(int32_t input_height) {
    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  inline int32_t InputHeight() const { return input_height_; }

  inline QuantizedConv2DTester& InputWidth(int32_t input_width) {
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

  inline QuantizedConv2DTester& KernelHeight(int32_t kernel_height) {
    EXPECT_GT(kernel_height, 0);
    kernel_height_ = kernel_height;
    return *this;
  }

  inline int32_t KernelHeight() const { return kernel_height_; }

  inline QuantizedConv2DTester& KernelWidth(int32_t kernel_width) {
    EXPECT_GT(kernel_width, 0);
    kernel_width_ = kernel_width;
    return *this;
  }

  inline int32_t KernelWidth() const { return kernel_width_; }

  inline QuantizedConv2DTester& StrideHeight(int32_t stride_height) {
    EXPECT_GT(stride_height, 0);
    stride_height_ = stride_height;
    return *this;
  }

  inline int32_t StrideHeight() const { return stride_height_; }

  inline QuantizedConv2DTester& StrideWidth(int32_t stride_width) {
    EXPECT_GT(stride_width, 0);
    stride_width_ = stride_width;
    return *this;
  }

  inline int32_t StrideWidth() const { return stride_width_; }

  inline QuantizedConv2DTester& DilationHeight(int32_t dilation_height) {
    EXPECT_GT(dilation_height, 0);
    dilation_height_ = dilation_height;
    return *this;
  }

  inline int32_t DilationHeight() const { return dilation_height_; }

  inline QuantizedConv2DTester& DilationWidth(int32_t dilation_width) {
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

  inline QuantizedConv2DTester& Groups(int32_t groups) {
    EXPECT_EQ(InputChannels() % groups, 0);
    EXPECT_EQ(OutputChannels() % groups, 0);
    groups_ = groups;
    return *this;
  }

  inline int32_t Groups() const { return groups_; }

  inline int32_t KernelInputChannels() const {
    return input_channels_ / groups_;
  }

  inline QuantizedConv2DTester& InputZeroPoint(int32_t input_zero_point) {
    input_zero_point_ = input_zero_point;
    return *this;
  }

  inline int32_t InputZeroPoint() const { return input_zero_point_; }

  inline QuantizedConv2DTester& OutputZeroPoint(int32_t output_zero_point) {
    output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int32_t OutputZeroPoint() const { return output_zero_point_; }

  inline QuantizedConv2DTester& KernelZeroPoint(int32_t kernel_zero_point) {
    kernel_zero_point_ = kernel_zero_point;
    return *this;
  }

  inline int32_t KernelZeroPoint() const { return kernel_zero_point_; }

  inline QuantizedConv2DTester& InputScale(float input_scale) {
    input_scale_ = input_scale;
    return *this;
  }

  inline float InputScale() const { return input_scale_; }

  inline QuantizedConv2DTester& KernelScale(float kernel_scale) {
    kernel_scale_ = kernel_scale;
    return *this;
  }

  inline float KernelScale() const {
    EXPECT_FALSE(ChannelWise());
    return kernel_scale_;
  }

  inline QuantizedConv2DTester& KernelScales(
      const std::vector<float>& kernel_scales) {
    EXPECT_GT(kernel_scales.size(), 0);
    kernel_scales_ = kernel_scales;
    return *this;
  }

  inline const std::vector<float>& KernelScales() const {
    EXPECT_TRUE(ChannelWise());
    return kernel_scales_;
  }

  inline bool Unsigned() const { return kernel_zero_point_ != 0; }

  inline bool ChannelWise() const { return !kernel_scales_.empty(); }

  inline QuantizedConv2DTester& OutputScale(float output_scale) {
    output_scale_ = output_scale;
    return *this;
  }

  inline float OutputScale() const { return output_scale_; }

  inline QuantizedConv2DTester& SamePadding() {
    padding_ = ::tflite::Padding_SAME;
    return *this;
  }

  inline QuantizedConv2DTester& ValidPadding() {
    padding_ = ::tflite::Padding_VALID;
    return *this;
  }

  inline QuantizedConv2DTester& ReluActivation() {
    activation_ = ::tflite::ActivationFunctionType_RELU;
    return *this;
  }

  inline QuantizedConv2DTester& Relu6Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU6;
    return *this;
  }

  inline QuantizedConv2DTester& ReluMinus1To1Activation() {
    activation_ = ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    return *this;
  }

  inline QuantizedConv2DTester& WeightsCache(
      TfLiteXNNPackDelegateWeightsCache* weights_cache) {
    weights_cache_ = weights_cache;
    return *this;
  }

  template <class T>
  void Test(Interpreter* delegate_interpreter,
            Interpreter* default_interpreter) const;

  void Test(TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel() const;

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
  int32_t input_zero_point_ = 0;
  int32_t output_zero_point_ = 0;
  int32_t kernel_zero_point_ = 0;
  float input_scale_ = 0.125f;
  float kernel_scale_ = 0.25f;
  std::vector<float> kernel_scales_;
  float output_scale_ = 1.5f;
  ::tflite::Padding padding_ = ::tflite::Padding_VALID;
  ::tflite::ActivationFunctionType activation_ =
      ::tflite::ActivationFunctionType_NONE;
  TfLiteXNNPackDelegateWeightsCache* weights_cache_ = nullptr;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_CONV_2D_TESTER_H_
