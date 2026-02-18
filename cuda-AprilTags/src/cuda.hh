#pragma once

#include <chrono>
#include <span>
#include <iostream>
#include <cassert>
#include <vector>

// ABSL replace
#define FATAL true
#define INFO true
#define LOG(...) std::cout
#define VLOG(...) std::cout
#define CHECK_EQ(x, y) { assert(x == y); }
#define CHECK(x) { assert(x); }
#define CHECK_LE(x, y) { assert(x <= y); }
#define CHECK_LT(x, y) { assert(x < y); }

#include <cuda_runtime.h>
#include <device_launch_parameters.h>



#define CHECK_CUDA(condition)                                             \
  if (auto c = condition)                                                 \
  LOG(FATAL) << "Check failed: " #condition " (" << cudaGetErrorString(c) \
             << ") "

namespace apriltag {

class CudaStream {
 public:
  CudaStream() { CHECK_CUDA(cudaStreamCreate(&stream_)); }

  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  virtual ~CudaStream() { CHECK_CUDA(cudaStreamDestroy(stream_)); }

  cudaStream_t get() { return stream_; }

 private:
  cudaStream_t stream_;
};

class CudaEvent {
 public:
  CudaEvent() { CHECK_CUDA(cudaEventCreate(&event_)); }

  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;

  virtual ~CudaEvent() { CHECK_CUDA(cudaEventDestroy(event_)); }

  void Record(CudaStream *stream) {
    CHECK_CUDA(cudaEventRecord(event_, stream->get()));
  }

  std::chrono::nanoseconds ElapsedTime(const CudaEvent &start) {
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start.event_, event_));
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<float, std::milli>(ms));
  }

  void Synchronize() { CHECK_CUDA(cudaEventSynchronize(event_)); }

 private:
  cudaEvent_t event_;
};

template <typename T>
class HostMemory {
 public:
  HostMemory(size_t size) {
    T *memory;
    CHECK_CUDA(cudaMallocHost((void **)(&memory), size * sizeof(T)));
    span_ = std::span<T>(memory, size);
  }
  HostMemory(const HostMemory &) = delete;
  HostMemory &operator=(const HostMemory &) = delete;

  virtual ~HostMemory() { CHECK_CUDA(cudaFreeHost(span_.data())); }

  T *get() { return span_.data(); }
  const T *get() const { return span_.data(); }

  size_t size() const { return span_.size(); }

  void MemcpyFrom(const T *other) {
    memcpy(span_.data(), other, sizeof(T) * size());
  }
  void MemcpyTo(const T *other) {
    memcpy(other, span_.data(), sizeof(T) * size());
  }

 private:
  std::span<T> span_;
};

template <typename T>
class GpuMemory {
 public:

  GpuMemory(size_t size) : size_(size) {
    CHECK_CUDA(cudaMalloc((void **)(&memory_), size * sizeof(T)));
  }
  GpuMemory(const GpuMemory &) = delete;
  GpuMemory &operator=(const GpuMemory &) = delete;

  virtual ~GpuMemory() { CHECK_CUDA(cudaFree(memory_)); }

  T *get() { return memory_; }
  const T *get() const { return memory_; }

  size_t size() const { return size_; }

  void MemcpyAsyncFrom(const T *host_memory, CudaStream *stream) {
    CHECK_CUDA(cudaMemcpyAsync(memory_, host_memory, sizeof(T) * size_,
                               cudaMemcpyHostToDevice, stream->get()));
  }
  void MemcpyAsyncFrom(const HostMemory<T> *host_memory, CudaStream *stream) {
    MemcpyAsyncFrom(host_memory->get(), stream);
  }

  void MemcpyAsyncTo(T *host_memory, size_t size, CudaStream *stream) const {
    CHECK_CUDA(cudaMemcpyAsync(reinterpret_cast<void *>(host_memory),
                               reinterpret_cast<void *>(memory_),
                               sizeof(T) * size, cudaMemcpyDeviceToHost,
                               stream->get()));
  }
  void MemcpyAsyncTo(T *host_memory, CudaStream *stream) const {
    MemcpyAsyncTo(host_memory, size_, stream);
  }
  void MemcpyAsyncTo(HostMemory<T> *host_memory, CudaStream *stream) const {
    MemcpyAsyncTo(host_memory->get(), stream);
  }

  void MemcpyFrom(const T *host_memory) {
    CHECK_CUDA(cudaMemcpy(reinterpret_cast<void *>(memory_),
                          reinterpret_cast<const void *>(host_memory),
                          sizeof(T) * size_, cudaMemcpyHostToDevice));
  }
  void MemcpyFrom(const HostMemory<T> *host_memory) {
    MemcpyFrom(host_memory->get());
  }

  void MemcpyTo(T *host_memory, size_t size) const {
    CHECK_CUDA(cudaMemcpy(reinterpret_cast<void *>(host_memory), memory_,
                          sizeof(T) * size, cudaMemcpyDeviceToHost));
  }
  void MemcpyTo(T *host_memory) const { MemcpyTo(host_memory, size_); }
  void MemcpyTo(HostMemory<T> *host_memory) const {
    MemcpyTo(host_memory->get());
  }

  void MemsetAsync(const uint8_t val, CudaStream *stream) const {
    CHECK_CUDA(cudaMemsetAsync(memory_, val, sizeof(T) * size_, stream->get()));
  }

  std::vector<T> Copy(size_t s) const {
    CHECK_LE(s, size_);
    std::vector<T> result(s);
    MemcpyTo(result.data(), s);
    return result;
  }

  std::vector<T> Copy() const { return Copy(size_); }

 private:
  T *memory_;
  const size_t size_;
};

void CheckAndSynchronize(std::string_view message = "");

void MaybeCheckAndSynchronize();
void MaybeCheckAndSynchronize(std::string_view message);

}  // namespace frc971::apriltag
