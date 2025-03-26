#pragma once
#ifndef GPUBUFFER
#define GPUBUFFER

#include<Eigen/Eigen>
#include<utils/gpumemory.cuh>

template <int DIMS>
__host__ __device__ inline uint32_t shape2size(Eigen::Vector<int, DIMS> &shape){
    uint32_t size = 1;
    for(int i=0; i<DIMS; i++){
        size*=shape[i];
    }
    return size;
}

template <typename T, int DIMS>
class GPUBuffer{
    public:
        Eigen::Vector<int, DIMS> m_shape;
        GPUVector<T> data_gpu;
        std::vector<T> data_cpu;
        cudaStream_t stream;

    public:
        GPUBuffer(){}
        GPUBuffer(Eigen::Vector<int, DIMS> shape, cudaStream_t s_ = nullptr): 
            m_shape(shape),
            data_gpu(shape2size<DIMS>(shape)),
            data_cpu(shape2size<DIMS>(shape)),
            stream(s_){}
        ~GPUBuffer(){}

        __host__ __device__ T * gpu(){return data_gpu.ptr();}
        __host__ __device__ T * cpu(){return &data_cpu[0];}

        void gpu2cpu(){data_gpu.memcpyto(cpu(), shape2size(m_shape), stream);}
        void cpu2gpu(){data_gpu.memcpyfrom(cpu(), shape2size(m_shape), stream);}

        uint32_t size() const{return shape2size(m_shape);}

};

#endif