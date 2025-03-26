#pragma once
#ifndef TRIANGLE_SAMPLER
#define TRIANGLE_SAMPLER

#include<geometry/triangle.cuh>
#include<geometry/boundingbox.cuh>
#include<utils/m_cuda_utils.cuh>
#include<utils/gpumemory.cuh>
#include<utils/random.cuh>
#include<utils/math_ex.cuh>
#include<vector>

inline __device__ __host__ uint32_t sample_idx(float prob, float *distri_int, uint32_t length){
    return binary_search(prob, distri_int, length);
}

__global__ void sample_uniform_on_triangle_kernel(
    Eigen::Vector3f *positions,
    uint32_t num,
    float *distri_int,
    uint32_t length,
    Triangle *triangles
){
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i>=num)return;

    Eigen::Vector3f sample = positions[i];
    uint32_t tri_idx = sample_idx(sample.x(), distri_int, length);

    positions[i] = triangles[tri_idx].sample_uniform_position(sample.tail<2>());
}

__global__ void sample_uniform_on_aabb_kernel(
    Eigen::Vector3f *positions,
    uint32_t num,
    BoundingBox aabb
){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= num)return;

    positions[idx] = aabb.min + positions[idx].cwiseProduct(aabb.diag());
}

__global__ void sample_perturbation_near_triangle_kernel(
    Eigen::Vector3f *positions,
    Eigen::Vector3f *perturb,
    uint32_t num
){
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= num)return;

    positions[i] = positions[i] + perturb[i];
}

inline std::vector<float> triangle_weights(std::vector<Triangle> &triangles){
    std::vector<float> weights(triangles.size());
    float total_area = 0.0f;
    for(size_t i=0; i < triangles.size(); i++){
        weights[i] = triangles[i].area();
        total_area += weights[i];
    }
    float inv_total_area = 1.0f / total_area;
    for(size_t i=0; i < triangles.size(); i++){
        weights[i] *= inv_total_area;
    }
    return weights;
}

inline std::vector<float> triangle_weights_integration(std::vector<float> &weights){
    std::vector<float> integrate(weights.size());
    float accumulate = 0.0f;
    for(size_t i=0; i < weights.size(); i++){
        accumulate += weights[i];
        integrate[i] = accumulate;
    }
    integrate.back() = 1.0f;
    return integrate;
}

inline std::vector<Eigen::Vector3f> sample_on_triangles(
    std::vector<Triangle> &triangles,
    size_t num_surface,
    size_t num_surface_offset,
    size_t num_uniform
){
    cudaStream_t stream;
    CUDA_CHECK_THROW(cudaStreamCreate(&stream));
    pcg32_random_t rng;

    GPUVector<Triangle> triangles_gpu;
    triangles_gpu.resize_and_memcpy_from_vector(triangles, stream);

    // generate the distribution for random sampling
    std::vector<float> distri = triangle_weights(triangles);    // distribuion
    std::vector<float> distri_i = triangle_weights_integration(distri); // distribution integration
    GPUVector<float> distri_i_gpu;
    GPUVector<Eigen::Vector3f> perturbation(num_surface_offset);
    distri_i_gpu.resize_and_memcpy_from_vector(distri_i, stream);


    BoundingBox aabb(Eigen::Vector3f::Constant(-1.0f), Eigen::Vector3f::Ones());
    uint32_t total_num = num_surface + num_surface_offset + num_uniform;
    GPUVector<Eigen::Vector3f> positions_gpu(total_num);
    std::vector<Eigen::Vector3f> positions_cpu(total_num);
    
    // all of this float number range from 0 to 1
    generate_random_uniform<float>((float*)positions_gpu.ptr(), total_num * 3, rng, 0.0f, 1.0f, stream);
    // The first float number would be seen as the probabilty of selection of triangles
    // The last two float number would be seen as the u,v coordinate on that triangle

    // sample on surface and near surface
    sample_uniform_on_triangle_kernel<<<div_round_up(num_surface + num_surface_offset, (size_t)128), 128, 0, stream>>>(
        positions_gpu.ptr(), 
        num_surface + num_surface_offset, 
        distri_i_gpu.ptr(),
        (uint32_t)distri_i_gpu.size(),
        triangles_gpu.ptr()
    );
    
    // sample near surface
    float stddev = 1.0f / 1024.0f;  
    generate_random_logistic<float>((float*)perturbation.ptr(), num_surface_offset*3, rng, 0.0f, stddev, stream);

    sample_perturbation_near_triangle_kernel<<<div_round_up(num_surface_offset, (size_t)128), 128, 0, stream>>>(
        positions_gpu.ptr() + num_surface,
        perturbation.ptr(),
        num_surface_offset
    );


    // sample uniform 
    sample_uniform_on_aabb_kernel<<<div_round_up(num_uniform, (size_t)128), 128, 0, stream>>>(
        positions_gpu.ptr() + num_surface + num_surface_offset,
        num_uniform,
        aabb
    );
    // assign float
    // I don't know the meaning of assign float
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    positions_gpu.memcpyto(positions_cpu.data(), positions_cpu.size(), stream);
    return positions_cpu;
}

inline std::vector<Eigen::Vector3f> sample_on_triangles(
    std::vector<Triangle> &triangles, 
    size_t num,
    Eigen::Vector3f weights = Eigen::Vector3f(1.0f, 0.0f, 0.0f)
){
    float sum = weights.x() + weights.y() + weights.z();
    weights /= sum;
    size_t surface_num = (size_t)(weights.x() * num);
    size_t near_num = (size_t)(weights.y() * num);
    size_t uniform_num = num - surface_num - near_num;
    return sample_on_triangles(triangles, surface_num, near_num, uniform_num);
}





#endif