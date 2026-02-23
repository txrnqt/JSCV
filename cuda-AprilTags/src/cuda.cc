#include "cuda.hh"

namespace apriltag {

size_t overall_memory = 0;

void CheckAndSynchronize(std::string_view message) {
  CHECK_CUDA(cudaDeviceSynchronize()) << message;
  CHECK_CUDA(cudaGetLastError()) << message;
}

void MaybeCheckAndSynchronize() {
  if (false) CheckAndSynchronize();
}

void MaybeCheckAndSynchronize(std::string_view message) {
  if (false) CheckAndSynchronize(message);
}

}
