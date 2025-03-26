#pragma once
#ifndef SPACE_SAMPLER
#define SPACE_SAMPLER

#include<Eigen/Eigen>
#include<utils/math_ex.cuh>

__host__ __device__ inline Eigen::Vector3i natural_to_integer(Eigen::Vector3i n, Eigen::Vector3i shape){
    return n - (shape/2);
}

__host__ __device__ inline Eigen::Vector3f voxel_size(Eigen::Vector3i shape){
    return Eigen::Vector3f(2.0f/shape.x(), 2.0f/shape.y(), 2.0f/shape.z());
}

__host__ __device__ inline Eigen::Vector3f voxel_center(Eigen::Vector3i p, Eigen::Vector3i shape){
    return Eigen::Vector3f(0.5f+p.x(), 0.5f+p.y(), 0.5f+p.z()).array() * voxel_size(shape).array();
}

__host__ __device__ inline Eigen::Vector3f thread_to_center(Eigen::Vector3i t, Eigen::Vector3i shape){
    Eigen::Vector3i p = t - (shape/2);
    return Eigen::Vector3f(0.5f+p.x(), 0.5f+p.y(), 0.5f+p.z()).array() * voxel_size(shape).array();
}

__global__ static void uniform_sample_voxel_center(Eigen::Vector3f *centers, Eigen::Vector3i shape){
    const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t j = threadIdx.y + blockDim.y * blockIdx.y;
    const uint32_t k = threadIdx.z + blockDim.z * blockIdx.z;
    if(i>=shape.x() || j>=shape.y() || k>=shape.z())return;
    size_t idx = i * shape.y() * shape.z() + j * shape.z() + k;
    
    Eigen::Vector3f f = thread_to_center({i, j, k}, shape);
    centers[idx] = f;
}

inline std::vector<Eigen::Vector3f> sample_voxel_center(Eigen::Vector3i shape){
    cudaStream_t stream;
    CUDA_CHECK_THROW(cudaStreamCreate(&stream));
    size_t num = shape.x() * shape.y() * shape.z();
    GPUVector<Eigen::Vector3f> centers_gpu(num);
    std::vector<Eigen::Vector3f> centers_cpu(num);

    const dim3 threads = {16, 16, 2};   // max threads of each block is 512
    const dim3 blocks = {div_round_up((uint32_t)shape.x(), 16u), div_round_up((uint32_t)shape.y(), 16u), div_round_up((uint32_t)shape.z(), 2u)};
    uniform_sample_voxel_center<<<blocks, threads, 0, stream>>>(centers_gpu.ptr(), shape);
    cudaError_t cudaStatus = cudaGetLastError();
    if(cudaStatus != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(cudaStatus));
    }
    cudaDeviceSynchronize();
    centers_gpu.memcpyto(&centers_cpu[0], num);
    return centers_cpu;
}

#endif