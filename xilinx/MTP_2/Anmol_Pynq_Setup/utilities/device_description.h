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

// Describes the underlying platform for a StreamExecutor; e.g. OpenCL or CUDA
// device and platform properties. Also contains convenience functions for
// checking/calculating launch dimensionality based on device properties.

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "str_join.h"
#include "str_split.h"
#include "device_description.pb.h"
#include "launch_dim.h"

namespace stream_executor {
namespace internal {
class DeviceDescriptionBuilder;
}  // namespace internal

// CUDA compute capability, as reported by the device description.
struct CudaComputeCapability {
  int major = 0;
  int minor = 0;

  // MSVC does not like "PASCAL" symbol.
  enum CudaComputeCapabilities {
    PASCAL_ = 6,
    VOLTA = 7,
    AMPERE = 8,
    HOPPER = 9
  };

  CudaComputeCapability() = default;
  CudaComputeCapability(int major, int minor) {
    this->major = major;
    this->minor = minor;
  }

  explicit CudaComputeCapability(const CudaComputeCapabilityProto &proto) {
    this->major = proto.major();
    this->minor = proto.minor();
  }

  bool IsAtLeast(int other_major, int other_minor = 0) const {
    return !(*this < CudaComputeCapability{other_major, other_minor});
  }

  bool operator<(const CudaComputeCapability &other) const {
    return ToPair() < other.ToPair();
  }

  bool operator==(const CudaComputeCapability &other) const {
    return ToPair() == other.ToPair();
  }

  bool operator!=(const CudaComputeCapability &other) const {
    return !(*this == other);
  }

  // Maximum resident blocks per multiprocessor, values taken from
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities.
  int GetMaxResidentBlocksPerSM() const {
    if (IsAtLeast(8, 6)) {
      return 16;
    } else if (IsAtLeast(8)) {
      return 32;
    } else if (IsAtLeast(7, 5)) {
      return 16;
    }
    return 32;
  }

  // Maximum resident warps per multiprocessor, values taken from
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities.
  int GetMaxResidentWarpsPerSM() const {
    if (IsAtLeast(8, 6)) {
      return 48;
    } else if (IsAtLeast(8)) {
      return 64;
    } else if (IsAtLeast(7, 5)) {
      return 32;
    }
    return 64;
  }

  std::string ToString() const { return absl::StrCat(major, ".", minor); }

  std::pair<int, int> ToPair() const { return std::make_pair(major, minor); }

  CudaComputeCapabilityProto ToProto() const {
    CudaComputeCapabilityProto proto;
    proto.set_major(major);
    proto.set_minor(minor);
    return proto;
  }
};

// ROCm compute capability, as reported by the device description.
class RocmComputeCapability {
 public:
  // gcn_arch_name example --  gfx90a:sramecc+:xnack-
  // gfx_version is the "gfx90a" part of the gcn_arch_name
  explicit RocmComputeCapability(const std::string &gcn_arch_name)
      : gcn_arch_name_(gcn_arch_name) {}

  explicit RocmComputeCapability(const RocmComputeCapabilityProto &proto)
      : gcn_arch_name_(proto.gcn_arch_name()) {}

  RocmComputeCapability() = default;
  ~RocmComputeCapability() = default;

  std::string gcn_arch_name() { return gcn_arch_name_; }

  std::string gfx_version() {
    std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name_, ':');
    return tokens[0];
  }

  bool is_supported_gfx_version() {
    return supported_gfx_versions().count(gfx_version()) != 0;
  }

  std::string supported_gfx_versions_str() {
    return absl::StrJoin(supported_gfx_versions(), ", ");
  }

  bool has_nhwc_layout_support() {
    return gfx_versions_with_nhwc_layout_support().count(gfx_version()) != 0;
  }

  bool has_bf16_dtype_support() {
    return gfx_versions_with_fast_bf16_support().count(gfx_version()) != 0;
  }

  bool has_fast_fp16_support() {
    return gfx_versions_with_fast_fp16_support().count(gfx_version()) != 0;
  }

  bool has_mfma_instr_support() {
    return gfx_versions_with_mfma_instr_support().count(gfx_version()) != 0;
  }

  bool has_fp16_atomics_support() {
    return gfx_versions_with_fp16_atomics_support().count(gfx_version()) != 0;
  }

  RocmComputeCapabilityProto ToProto() const {
    RocmComputeCapabilityProto proto;
    proto.set_gcn_arch_name(gcn_arch_name_);
    return proto;
  }

 private:
  std::string gcn_arch_name_;
  std::set<std::string> supported_gfx_versions() {
    return {
        "gfx900",  // MI25
        "gfx906",  // MI50 / MI60
        "gfx908",  // MI100
        "gfx90a",  // MI200
        "gfx1030"  // Navi21
    };
  }
  std::set<std::string> gfx_versions_with_nhwc_layout_support() {
    return {"gfx908", "gfx90a"};
  }
  std::set<std::string> gfx_versions_with_fast_bf16_support() {
    return {"gfx908", "gfx90a"};
  }
  std::set<std::string> gfx_versions_with_fast_fp16_support() {
    return {"gfx906", "gfx908", "gfx90a", "gfx1030"};
  }
  std::set<std::string> gfx_versions_with_mfma_instr_support() {
    return {"gfx908", "gfx90a"};
  }
  std::set<std::string> gfx_versions_with_fp16_atomics_support() {
    // TODO(rocm): Check. This should be the same as
    // gfx_versions_with_fast_fp16_support.
    return {"gfx90a"};
  }
};

// Data that describes the execution target of the StreamExecutor, in terms of
// important logical parameters. These include dimensionality limits and
// physical parameters of interest, such as number of cores present on the
// device.
//
// Thread-safe: immutable post-initialization.
class DeviceDescription {
 public:
  // Returns the platform being run on; this value is primarily intended for
  // printing, and comes out something like "OpenCL 1.2" or "Compute Capability
  // 3.5".
  const std::string &platform_version() const { return platform_version_; }

  // Returns the driver version interfacing with the underlying platform. Vendor
  // dependent format.
  const std::string &driver_version() const { return driver_version_; }

  // Return the runtime version, if one is provided by the underlying platform.
  // Vendor dependent format / usefulness.
  const std::string &runtime_version() const { return runtime_version_; }

  // Returns the name that the device reports. Vendor dependent.
  const std::string &name() const { return name_; }

  // Gets a human-readable description of the device, e.g. "nvidia GPU
  // supporting sm75 with 32GB RAM, 80 SMs, ...".  This is intended to be the
  // same if and only if two devices are "the same" (e.g. the same make/model of
  // GPU), though it may not completely succeed at this for all platforms.
  //
  // This string is not guaranteed to be stable between versions.  Please DO NOT
  // rely on it never changing.  (Within one version of the code, it won't
  // change, don't worry.)
  const std::string &model_str() const { return model_str_; }

  // Returns the PCI bus identifier for this device, of the form
  // [domain]:[bus]:[device].[function]
  const std::string &pci_bus_id() const { return pci_bus_id_; }

  // Returns the NUMA node associated with this device, for use in
  // determining socket locality. If the NUMA node could not be determined, -1
  // is returned.
  int numa_node() const { return numa_node_; }

  // Number of cores (traditional notion of core; i.e. an SM on an NVIDIA device
  // or an AMD Compute Unit.
  int core_count() const { return core_count_; }

  // Number of floating point operations one core (SM, compute unit) can execute
  // in parallel. Corresponds to the number of "CUDA cores" for NVIDIA devices.
  int fpus_per_core() const { return fpus_per_core_; }

  // Returns the limit on the thread dimensionality values in each of the
  // respective dimensions. These limits affect what constitutes a legitimate
  // kernel launch request.
  const ThreadDim &thread_dim_limit() const { return thread_dim_limit_; }

  // Returns the limit on the block dimensionality values in each of the
  // respective dimensions. These limits may affect what constitutes a
  // legitimate kernel launch request.
  const BlockDim &block_dim_limit() const { return block_dim_limit_; }

  // Returns the limit on the total number of threads that can be launched in a
  // single block; i.e. the limit on x * y * z dimensions of a ThreadDim.
  // This limit affects what constitutes a legitimate kernel launch request.
  const int64_t &threads_per_block_limit() const {
    return threads_per_block_limit_;
  }

  // Returns the limit on the total number of threads that can be simultaneously
  // launched on a given multiprocessor.
  const int64_t &threads_per_core_limit() const {
    return threads_per_core_limit_;
  }

  // Returns the number of threads per warp/wavefront.
  const int64_t &threads_per_warp() const { return threads_per_warp_; }

  // Returns the limit on the total number of registers per core.
  const int64_t &registers_per_core_limit() const {
    return registers_per_core_limit_;
  }

  // Returns the limit on the total number of registers that can be
  // simultaneously used by a block.
  const int64_t &registers_per_block_limit() const {
    return registers_per_block_limit_;
  }

  // Returns the number of address bits available to kernel code running on the
  // platform. This affects things like the maximum allocation size and perhaps
  // types used in kernel code such as size_t.
  const int64_t &device_address_bits() const { return device_address_bits_; }

  // Returns the device memory size in bytes.
  int64_t device_memory_size() const { return device_memory_size_; }

  // Returns the L2 cache size in bytes.
  int64_t l2_cache_size() const { return l2_cache_size_; }

  // Returns the device's memory bandwidth in bytes/sec.  (This is for
  // reads/writes to/from the device's own memory, not for transfers between the
  // host and device.)
  int64_t memory_bandwidth() const { return memory_bandwidth_; }

  // Returns the device's core clock rate in GHz.
  float clock_rate_ghz() const { return clock_rate_ghz_; }

  // Returns whether ECC is enabled.
  bool ecc_enabled() const { return ecc_enabled_; }

  // Returns the device vendor string, e.g., "NVIDIA Corporation", "Advanced
  // Micro Devices, Inc.", or "GenuineIntel".
  const std::string &device_vendor() const { return device_vendor_; }

  // Returns the CUDA compute capability if we're running on the CUDA platform.
  // If a CUDA compute capability is not available, the major version will be
  // zero.
  CudaComputeCapability cuda_compute_capability() const;

  // Returns the ROCm compute capability if we're running on the ROCm platform.
  // If a ROCm compute capability is not available, the default gfx_arch will
  // be "gfx000" (which is an invalid gfx arch).
  RocmComputeCapability rocm_compute_capability() const;

  // Returns the maximum amount of shared memory present on a single core
  // (i.e. Streaming Multiprocessor on NVIDIA GPUs; Compute Unit for OpenCL
  // devices). Note that some devices, such as NVIDIA's have a configurable
  // partitioning between shared memory and L1 cache.
  int64_t shared_memory_per_core() const { return shared_memory_per_core_; }

  // Returns the maximum amount of static shared memory
  // available for a single block.
  int64_t shared_memory_per_block() const { return shared_memory_per_block_; }

  // Returns the maximum amount of shared memory available for a single block
  // including the dynamically allocated one.
  int64_t shared_memory_per_block_optin() const {
    return shared_memory_per_block_optin_;
  }

  // TODO(leary): resident blocks per core will be useful.

  // Convenience typedef for the string-based DeviceDescription mapping.
  typedef std::map<std::string, std::string> Map;

  // Returns a mapping from readable names to readable values that describe the
  // device. This is useful for things like printing.
  std::unique_ptr<Map> ToMap() const;

  // For string values that are not available via the underlying platform, this
  // value will be provided.
  static const char *kUndefinedString;

 private:
  friend class internal::DeviceDescriptionBuilder;

  DeviceDescription();

  // For description of the following members, see the corresponding accessor
  // above.
  //
  // N.B. If another field is added, update ToMap() above.
  std::string device_vendor_;
  std::string platform_version_;
  std::string driver_version_;
  std::string runtime_version_;
  std::string pci_bus_id_;
  std::string name_;
  std::string model_str_;

  ThreadDim thread_dim_limit_;
  BlockDim block_dim_limit_;

  int64_t threads_per_core_limit_;
  int64_t threads_per_block_limit_;
  int64_t threads_per_warp_;

  int64_t registers_per_core_limit_;
  int64_t registers_per_block_limit_;

  int64_t device_address_bits_;
  int64_t device_memory_size_;
  int64_t l2_cache_size_;
  int64_t memory_bandwidth_;

  // Shared memory limits on a given device.
  int64_t shared_memory_per_core_;
  int64_t shared_memory_per_block_;
  int64_t shared_memory_per_block_optin_;

  float clock_rate_ghz_;

  // CUDA "CC" major value, -1 if not available.
  CudaComputeCapability cuda_compute_capability_{-1, -1};

  // ROCm gfx arch,  "gfx000" if not available.
  RocmComputeCapability rocm_compute_capability_{"gfx000"};

  int numa_node_;
  int core_count_;
  int fpus_per_core_;
  bool ecc_enabled_;

  SE_DISALLOW_COPY_AND_ASSIGN(DeviceDescription);
};

namespace internal {

// Helper class the builds a device description, given that it has a large
// number of fields that would be easily confused in constructor form.
class DeviceDescriptionBuilder {
 public:
  DeviceDescriptionBuilder();

  // For descriptions of the following fields, see comments on the corresponding
  // DeviceDescription::* accessors above.

  void set_device_vendor(const std::string &value) {
    device_description_->device_vendor_ = value;
  }
  void set_platform_version(const std::string &value) {
    device_description_->platform_version_ = value;
  }
  void set_driver_version(const std::string &value) {
    device_description_->driver_version_ = value;
  }
  void set_runtime_version(const std::string &value) {
    device_description_->runtime_version_ = value;
  }
  void set_pci_bus_id(const std::string &value) {
    device_description_->pci_bus_id_ = value;
  }
  void set_name(const std::string &value) {
    device_description_->name_ = value;
  }
  void set_model_str(const std::string &value) {
    device_description_->model_str_ = value;
  }

  void set_thread_dim_limit(const ThreadDim &value) {
    device_description_->thread_dim_limit_ = value;
  }
  void set_block_dim_limit(const BlockDim &value) {
    device_description_->block_dim_limit_ = value;
  }

  void set_threads_per_core_limit(int64_t value) {
    device_description_->threads_per_core_limit_ = value;
  }
  void set_threads_per_block_limit(int64_t value) {
    device_description_->threads_per_block_limit_ = value;
  }
  void set_threads_per_warp(int64_t value) {
    device_description_->threads_per_warp_ = value;
  }

  void set_registers_per_core_limit(int64_t value) {
    device_description_->registers_per_core_limit_ = value;
  }
  void set_registers_per_block_limit(int64_t value) {
    device_description_->registers_per_block_limit_ = value;
  }

  void set_device_address_bits(int64_t value) {
    device_description_->device_address_bits_ = value;
  }
  void set_device_memory_size(int64_t value) {
    device_description_->device_memory_size_ = value;
  }
  void set_l2_cache_size(int64_t value) {
    device_description_->l2_cache_size_ = value;
  }
  void set_memory_bandwidth(int64_t value) {
    device_description_->memory_bandwidth_ = value;
  }

  void set_shared_memory_per_core(int64_t value) {
    device_description_->shared_memory_per_core_ = value;
  }
  void set_shared_memory_per_block(int64_t value) {
    device_description_->shared_memory_per_block_ = value;
  }
  void set_shared_memory_per_block_optin(int64_t value) {
    device_description_->shared_memory_per_block_optin_ = value;
  }

  void set_clock_rate_ghz(float value) {
    device_description_->clock_rate_ghz_ = value;
  }

  void set_cuda_compute_capability(int major, int minor) {
    device_description_->cuda_compute_capability_ =
        CudaComputeCapability{major, minor};
  }

  void set_rocm_compute_capability(std::string gcn_arch_name) {
    device_description_->rocm_compute_capability_ =
        RocmComputeCapability(gcn_arch_name);
  }

  void set_numa_node(int value) { device_description_->numa_node_ = value; }
  void set_core_count(int value) { device_description_->core_count_ = value; }
  void set_fpus_per_core(int value) {
    device_description_->fpus_per_core_ = value;
  }
  void set_ecc_enabled(bool value) {
    device_description_->ecc_enabled_ = value;
  }

  // Returns a built DeviceDescription with ownership transferred to the
  // caller. There are currently no restrictions on which fields must be set in
  // order to build the descriptor.
  //
  // Once the description is built, this builder object should be discarded.
  std::unique_ptr<DeviceDescription> Build() {
    return std::move(device_description_);
  }

 private:
  std::unique_ptr<DeviceDescription> device_description_;

  SE_DISALLOW_COPY_AND_ASSIGN(DeviceDescriptionBuilder);
};

}  // namespace internal

// Returns whether the given thread_dim is acceptable given the limits described
// in device_description. For detailed reasons for failing the predicate, enable
// VLOG(2) for this module.
bool ThreadDimOk(const DeviceDescription &device_description,
                 const ThreadDim &thread_dim);

// Calculate the number of threads/blocks required to process element_count
// elements. Note that you can still end up with more threads than
// element_count due to rounding, so kernels often start with an "is this
// thread id in the element_count range?" test.
void CalculateDimensionality(const DeviceDescription &device_description,
                             int64_t element_count, int64_t *threads_per_block,
                             int64_t *block_count);

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
