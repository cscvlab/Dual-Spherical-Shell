#pragma once
#ifndef RAY_TRACER_UTILS_H
#define RAY_TRACER_UTILS_H

#include<renderer/renderer.cuh>

class RayTracerUtils{
    private:
        std::vector<Triangle> triangles_cpu;
        GPUVector<Triangle> triangles_gpu;
        std::unique_ptr<TriangleBVH> bvh;
        SphereTracer tracer;
        cudaStream_t stream;
    
    public:
        RayTracerUtils(){
            CUDA_CHECK_THROW(cudaStreamCreate(&stream));
        }
        RayTracerUtils(std::string mesh_path){
            CUDA_CHECK_THROW(cudaStreamCreate(&stream));
            load_mesh(mesh_path);
        }
        RayTracerUtils(std::vector<Triangle> &triangles){
            CUDA_CHECK_THROW(cudaStreamCreate(&stream));
            load_mesh(triangles);
        }
        ~RayTracerUtils(){}

        void load_mesh(std::string mesh_path);
        void load_mesh(std::vector<Triangle> &triangles);

        std::vector<float> signed_distance(std::vector<Eigen::Vector3f> &positions, SDFCalcMode mode);
        std::vector<Eigen::Vector3f> trace(std::vector<Eigen::Vector3f> &origins, std::vector<Eigen::Vector3f> &directions);

        
};

#endif