#pragma once
#ifndef SDFGENERATOR
#define SDFGENERATOR

#include<cuda_runtime.h>

#include<time.h>

#include<geometry/trianglebvh.cuh>
#include<geometry/mesh.h>
#include<utils/gpumemory.cuh>



constexpr unsigned int N_PRIMITIVES_PER_NODE = 8;
class SDFGenerator{
    private:
        std::vector<Triangle> triangles_cpu;
        GPUVector<Triangle> triangles_gpu;
        std::shared_ptr<TriangleBVH> bvh;
        cudaStream_t stream;
    public:
        SDFGenerator(){
            CUDA_CHECK_THROW(cudaStreamCreate(&stream));
        }
        ~SDFGenerator(){}
        void load_obj(std::string path){
            // load obj
            Mesh::load_triangles(path, triangles_cpu);

            // generate bvh
            bvh = TriangleBVH::create();

            clock_t start, end;
            start = clock();
            bvh->build(triangles_cpu, N_PRIMITIVES_PER_NODE);
            end = clock();
            std::cout << "Building Triangle BVH Cost " << (float(end - start)/CLOCKS_PER_SEC) << " s." << std::endl;

            triangles_gpu.resize(triangles_cpu.size());
            triangles_gpu.memcpyfrom(&triangles_cpu[0], triangles_cpu.size(), stream);
            bvh->build_optix(triangles_gpu, stream);
        }
        std::vector<float> generate_sdf(std::vector<Eigen::Vector3f> &positions, SDFCalcMode mode){
            // prepare data and space
            GPUVector<Eigen::Vector3f> positions_gpu(positions.size());
            std::vector<float> distances_cpu(positions.size());
            GPUVector<float> distances_gpu(positions.size());
            positions_gpu.memcpyfrom(&positions[0], positions.size(), stream);
            std::cout << "Start Calculating" << std::endl;
            clock_t start, end;
            start = clock();
            bvh->signed_distance_gpu(positions_gpu.ptr(), distances_gpu.ptr(), positions.size(), triangles_gpu.ptr(), mode, false, stream);
            cudaDeviceSynchronize();
            end = clock();
            std::cout << "Calculate SDF Cost " << (float(end - start)/CLOCKS_PER_SEC) << "s." << std::endl;
            distances_gpu.memcpyto(&distances_cpu[0], positions.size(), stream);
            return distances_cpu;
        }
};



#endif
