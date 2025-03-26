#include<ray_tracer_utils.cuh>

void RayTracerUtils::load_mesh(std::string path){
    Mesh::load_triangles(path, triangles_cpu);

    bvh = TriangleBVH::create();
    bvh->build(triangles_cpu, 8);

    triangles_gpu.resize_and_memcpy_from_vector(triangles_cpu);
    bvh->build_optix(triangles_gpu, stream);
}

void RayTracerUtils::load_mesh(std::vector<Triangle> &triangles){
    triangles_cpu.resize(triangles.size());
    for(int i=0; i<triangles.size(); i++){
        triangles_cpu[i] = triangles[i];
    }

    bvh = TriangleBVH::create();
    bvh->build(triangles_cpu, 8);

    triangles_gpu.resize_and_memcpy_from_vector(triangles_cpu);
    bvh->build_optix(triangles_gpu, stream);
}

std::vector<float> RayTracerUtils::signed_distance(std::vector<Eigen::Vector3f> &positions, SDFCalcMode mode){
    GPUVector<Eigen::Vector3f> positions_gpu(positions.size());
    std::vector<float> distances_cpu(positions.size());
    GPUVector<float> distances_gpu(positions.size());
    positions_gpu.memcpyfrom(&positions[0], positions.size(), stream);
    clock_t start, end;
    start = clock();
    bvh->signed_distance_gpu(positions_gpu.ptr(), distances_gpu.ptr(), positions.size(), triangles_gpu.ptr(), mode, false, stream);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "Calculate SDF Cost " << (float(end - start)/CLOCKS_PER_SEC) << "s." << std::endl;
    distances_gpu.memcpyto(&distances_cpu[0], positions.size(), stream);
    return distances_cpu;
}

std::vector<Eigen::Vector3f> RayTracerUtils::trace(std::vector<Eigen::Vector3f> &positions, std::vector<Eigen::Vector3f> &directions){
    GPUVector<Eigen::Vector3f> positions_gpu;
    GPUVector<Eigen::Vector3f> directions_gpu;
    std::vector<Eigen::Vector3f> res(positions.size());
    positions_gpu.resize_and_memcpy_from_vector(positions, stream);
    directions_gpu.resize_and_memcpy_from_vector(directions, stream);

    bvh->ray_trace_gpu(positions_gpu.ptr(), directions_gpu.ptr(), positions.size(), triangles_gpu.ptr(), stream);
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    positions_gpu.memcpyto(res.data(), positions.size(), stream);
    return res;
}
