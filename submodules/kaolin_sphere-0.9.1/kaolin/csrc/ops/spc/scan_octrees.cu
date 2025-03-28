// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }

#include <cub/device/device_scan.cuh>

#include <ATen/ATen.h>

#include "../../spc_math.h"
#include "../../utils.h"

#define THREADS_PER_BLOCK 64

namespace kaolin {

using namespace cub;

__global__ void d_ScanNodesX(
    const uint numBytes,
    const uint8_t *d_octree,
    uint *octrees_ptr) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < numBytes)
    octrees_ptr[tidx] = __popc(d_octree[tidx]);
}

size_t GetScanOctreesTmpStorageBytes(at::Tensor num_childrens_per_node,
                                     at::Tensor prefix_sum, int max_total_points) {
  size_t temp_storage_bytes = 0;
  void* d_tmp = NULL;
  CubDebugExit(DeviceScan::InclusiveSum(
      d_tmp, temp_storage_bytes,
      reinterpret_cast<uint*>(num_childrens_per_node.data_ptr<int>()),
      reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()),
      max_total_points));
  return temp_storage_bytes;
}

int scan_octrees_cuda_kernel_launcher(
    at::Tensor octrees,
    at::Tensor lengths,
    at::Tensor num_childrens_per_node,
    at::Tensor prefix_sum,
    at::Tensor pyramid,
    at::Tensor temp_storage) {
  int batch_size = lengths.size(0);
  // get tensor data pointers
  uint8_t* octrees_ptr = octrees.data_ptr<uint8_t>();
  uint* num_childrens_per_node_ptr = reinterpret_cast<uint*>(num_childrens_per_node.data_ptr<int>());
  uint* prefix_sum_ptr = reinterpret_cast<uint*>(prefix_sum.data_ptr<int>());
  int* pyramid_ptr = pyramid.data_ptr<int>();

  void* d_temp_storage = (void*) temp_storage.data_ptr<uint8_t>();
  size_t temp_storage_bytes = temp_storage.size(0);

  uint* EX0 = prefix_sum_ptr;
  uint8_t* O0 = octrees_ptr;
  int* h0 = pyramid_ptr;
  int level;

  for (int batch = 0; batch < batch_size; batch++) {
    uint8_t*  O = O0;
    uint*   S = EX0 + 1;
    uint  osize = lengths[batch].item<int>();

    // compute exclusive sum 1 element beyond end of list to get inclusive sum starting at prefix_sum_ptr+1
    d_ScanNodesX<<< (osize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(
        osize, O0, num_childrens_per_node_ptr);
    CubDebugExit(DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, num_childrens_per_node_ptr,
        EX0, osize + 1)); // carful with the +1

    int* Pmid = h0;
    int* PmidSum = h0 + KAOLIN_SPC_MAX_LEVELS + 2;

    int Lsize = 1;
    uint currSum, prevSum = 0;

    uint sum = Pmid[0] = Lsize;
    PmidSum[0] = 0;
    PmidSum[1] = Lsize;

    level = 0;
    while (sum <= osize) {
      O += Lsize;
      S += Lsize;

      cudaMemcpy(&currSum, EX0 + prevSum + 1, sizeof(uint), cudaMemcpyDeviceToHost);
      CUDA_CHECK(cudaGetLastError());

      Lsize = currSum - prevSum;
      prevSum = currSum;

      Pmid[++level] = Lsize;
      sum += Lsize;
      PmidSum[level + 1] = sum;
    }

    O0 += osize;
    EX0 += (osize + 1);
    h0 += 2 * (KAOLIN_SPC_MAX_LEVELS + 2);
  }
  CUDA_CHECK(cudaGetLastError());

  return level;
}

}  // namespace kaolin
