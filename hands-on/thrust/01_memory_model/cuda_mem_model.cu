// C++ standard headers
#include <cassert>
#include <iostream>
#include <ranges>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// CUDA headers
#include <cuda_runtime.h>

// local headers
#include "cuda_check.h"

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main() {
  // Choose one CUDA device
  CUDA_CHECK(cudaSetDevice(MYDEVICE));

  // Pointer and dimension for host memory
  int dim = 1024;
  // Part 1: allocate a buffer in host memory with increasing values from 0 to
  // dimA - 1
  auto h_buffer = ...;

  // Part 2: allocate two buffers in device memory
  auto d_buffer1 = ...;
  auto d_buffer2 = ...;

  // Part 3: copy the content of the host buffer to the first device buffer

  // Part 4: copy the content of the first device buffer to the second device
  // buffer

  // Part 5: set all the values in the host buffer to zero

  // Part 6: copy the contant back to the host buffer

  // Verify the data on the host is correct
  assert(std::ranges::equal(h_buffer, std::views::iota(0, dim)));

  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  std::cout << "Correct!" << std::endl;

  return 0;
}
