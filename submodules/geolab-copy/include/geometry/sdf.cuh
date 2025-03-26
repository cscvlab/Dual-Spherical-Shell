#pragma once
#ifndef SDF
#define SDF

#include<Eigen/Eigen>
#include<vector>
#include<utils/gpumemory.cuh>

struct SDFPayload{
    Eigen::Vector3f dir;
    uint32_t idx;
    uint16_t n_steps;
    bool alive;
};

struct RaysSDFSoa{

    void resize(size_t s){
        pos.resize(s);
        nor.resize(s);
        distance.resize(s);
        prev_distance.resize(s);
        total_distance.resize(s);
        min_visibility.resize(s);
        payload.resize(s);
    }

    void memcpy_from(RaysSDFSoa &other, uint32_t num, cudaStream_t stream){
        CUDA_CHECK_THROW(cudaMemcpyAsync(pos.ptr(), other.pos.ptr(), num, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(nor.ptr(), other.nor.ptr(), num, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(distance.ptr(), other.distance.ptr(), num, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(prev_distance.ptr(), other.prev_distance.ptr(), num, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(total_distance.ptr(), other.total_distance.ptr(), num, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(min_visibility.ptr(), other.min_visibility.ptr(), num, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(payload.ptr(), other.payload.ptr(), num, cudaMemcpyDeviceToDevice, stream));
    }

    GPUVector<Eigen::Vector3f> pos;
    GPUVector<Eigen::Vector3f> nor;
    GPUVector<float> distance;
    GPUVector<float> prev_distance;
    GPUVector<float> total_distance;
    GPUVector<float> min_visibility;
    GPUVector<SDFPayload> payload;
};


#endif