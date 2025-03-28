#pragma once
#ifndef GPUMEM
#define GPUMEM

#include<utils/m_cuda_utils.cuh>
#include<utils/common.cuh>
#include<vector>

#ifdef DEBUG_LOG
#include<iostream>
#endif

template<typename T>
class GPUVector{
    private:
        T *data = nullptr;
        size_t m_size = 0;
    public:
        GPUVector(){}
        GPUVector(const size_t size){ resize(size);}
        ~GPUVector(){ free_memory(); }

        size_t size() const{return m_size;}
        T* ptr(){return data;}

        void memcpyfrom(T *hostptr, size_t size, cudaStream_t stream = nullptr){
            CUDA_CHECK_THROW(cudaMemcpyAsync(data, hostptr, size * sizeof(T), cudaMemcpyHostToDevice, stream));
        }

        void memcpyto(T *hostptr, size_t size, cudaStream_t stream = nullptr){
            CUDA_CHECK_THROW(cudaMemcpyAsync(hostptr, data, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
        }

        void memcpyfrom_device(T *devicePtr, size_t size, cudaStream_t stream = nullptr){
            CUDA_CHECK_THROW(cudaMemcpyAsync(devicePtr, data, size*sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }

        void memcpyto_device(T *devicePtr, size_t size, cudaStream_t stream = nullptr){
            CUDA_CHECK_THROW(cudaMemcpyAsync(data, devicePtr, size*sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }

        void resize(const size_t size){
            if (m_size != size) {
                if (m_size) {
                    try {
                        free_memory();
                    } catch (std::runtime_error error) {
                        throw std::runtime_error(std::string("Could not free memory: ") + error.what());
                    }
                }

                if (size > 0) {
                    try {
                        CUDA_CHECK_THROW(cudaMalloc((void**)&data, size * sizeof(T)));
                    } catch (std::runtime_error error) {
                        throw std::runtime_error(std::string("Could not allocate memory: ") + error.what());
                    }
                }
                m_size = size;
#ifdef DEBUG_LOG
                std::cout << "Allocate Memory: " << bytes_to_string(m_size * sizeof(T)) << std::endl;
#endif
            }
        }

        void free_memory() {
            if (!data)return;
            CUDA_CHECK_THROW(cudaFree(data));
            data = nullptr;
            m_size = 0;
#ifdef DEBUG_LOG
                std::cout << "Free Memory: " << bytes_to_string(m_size * sizeof(T)) << std::endl;
#endif
        }

        void resize_and_memcpy_from_vector(std::vector<T> &host, cudaStream_t stream = nullptr){
            resize(host.size());
            memcpyfrom(&host[0], host.size(), stream);
        }
};

template<typename T, int MAX_SIZE=32>
class FixedStack{
    public:
        __host__ __device__ void push(T value){
            if(m_size >= MAX_SIZE - 1){
                return;
            }
            m_elements[m_size++] = value;
        }

        __host__ __device__ T pop(){
            return m_elements[--m_size];
        }

        __host__ __device__ bool empty() const{
            return m_size <= 0;
        }

    private:
        T m_elements[MAX_SIZE];
        int m_size = 0;
};

using FixedIntStack = FixedStack<int>;


#endif 