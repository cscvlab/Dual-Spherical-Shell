#include<geometry/voxel.cuh>
#include<geometry/tribox3.h>

#include<utils/gpubuffer.cuh>
#include<time.h>
#include<queue>

__global__ void classify_voxels(float *distances, char *voxels, size_t num, float voxel_radius){
    const size_t idx = threadIdx.x * blockDim.x * blockIdx.x;
    if(idx >= num)return;

    float distance = distances[idx];
    if(std::abs(distance) < voxel_radius){
        voxels[idx] = SURFACE;
    }else{
        if(distance < 0){
            voxels[idx] = INSIDE;
        }else{
            voxels[idx] = OUTSIDE;
        }
    }
}

void DualVoxelDynamic::load_model(std::string obj_path){
    // 1. Load obj
    Mesh::load_triangles(obj_path, triangles_cpu);

    // 2. create bvh
    bvh = TriangleBVH::create();
    clock_t start, end;
    start = clock();
    bvh->build(triangles_cpu, 8);
    end = clock();
    std::cout << "Building Triangle BVH Cost " << (float(end - start)/CLOCKS_PER_SEC) << " s." << std::endl;
    triangles_gpu.resize_and_memcpy_from_vector(triangles_cpu, stream);
    bvh->build_optix(triangles_gpu, stream);
}

void DualVoxelDynamic::load_model(std::vector<Triangle> &triangles){
    // 1. Copy data to local memory
    triangles_cpu.resize(triangles.size());
    memcpy(&triangles_cpu[0], &triangles[0], triangles_cpu.size()*sizeof(Triangle));

    // 2. create bvh
    bvh = TriangleBVH::create();
    clock_t start, end;
    start = clock();
    bvh->build(triangles_cpu, 8);
    end = clock();
    std::cout << "Building Triangle BVH Cost " << (float(end - start)/CLOCKS_PER_SEC) << " s." << std::endl;
    triangles_gpu.resize_and_memcpy_from_vector(triangles_cpu, stream);
    bvh->build_optix(triangles_gpu, stream);
}

std::vector<Eigen::Vector3i> DualVoxelDynamic::surface_voxel(Eigen::Vector3i shape){
    std::vector<Eigen::Vector3i> surface_voxels;
    size_t voxel_num = shape.x() * shape.y() * shape.z();
    GPUVector<Eigen::Vector3f> voxel_centers(voxel_num);
    GPUVector<float> distances(voxel_num);
    GPUVector<char> tags_gpu(voxel_num);
    std::vector<char> tags_cpu(voxel_num);

    const dim3 threads = {16, 16, 2};
    const dim3 blocks = {div_round_up((uint32_t)shape.x(), 16u), div_round_up((uint32_t)shape.y(), 16u), div_round_up((uint32_t)shape.z(), 2u)};

    uniform_sample_voxel_center<<<blocks, threads, 0, stream>>>( voxel_centers.ptr(), shape);
    bvh->signed_distance_gpu(voxel_centers.ptr(), distances.ptr(), voxel_num, triangles_gpu.ptr(), m_mode, false, stream);

    classify_voxels<<<div_round_up(voxel_num, (size_t)128), 128, 0, stream>>>(distances.ptr(), tags_gpu.ptr(), voxel_num, voxel_size(shape).minCoeff()/2);

    tags_gpu.memcpyto(&tags_cpu[0], voxel_num, stream);

    Eigen::Vector3i min = -(shape/2);
    Eigen::Vector3i max = shape/2 - Eigen::Vector3i::Ones();

    for(int x=min.x(); x<=max.y(); x++){
        for(int y=min.y(); y<=max.y(); y++){
            for(int z=min.z(); z<=max.z(); z++){
                size_t idx = (x + shape.x()/2) * shape.y() * shape.z() + (y + shape.y()/2) * shape.z() + (z + shape.z()/2);
                if(tags_cpu[idx] == SURFACE){
                    surface_voxels.emplace_back(Eigen::Vector3i(x, y, z));
                }
            }
        }
    }

    return surface_voxels;
}

std::vector<Eigen::Vector3i> DualVoxelDynamic::inside_voxel(Eigen::Vector3i shape){
    std::vector<Eigen::Vector3i> inside_voxels;
    size_t voxel_num = shape.x() * shape.y() * shape.z();
    GPUVector<Eigen::Vector3f> voxel_centers(voxel_num);
    GPUVector<float> distances(voxel_num);
    GPUVector<char> tags_gpu(voxel_num);
    std::vector<char> tags_cpu(voxel_num);

    const dim3 threads = {16, 16, 2};
    const dim3 blocks = {div_round_up((uint32_t)shape.x(), 16u), div_round_up((uint32_t)shape.y(), 16u), div_round_up((uint32_t)shape.z(), 2u)};

    uniform_sample_voxel_center<<<blocks, threads, 0, stream>>>( voxel_centers.ptr(), shape);
    bvh->signed_distance_gpu(voxel_centers.ptr(), distances.ptr(), voxel_num, triangles_gpu.ptr(), m_mode, false, stream);

    classify_voxels<<<div_round_up(voxel_num, (size_t)128), 128, 0, stream>>>(distances.ptr(), tags_gpu.ptr(), voxel_num, voxel_size(shape).minCoeff()/2);

    tags_gpu.memcpyto(&tags_cpu[0], voxel_num, stream);

    Eigen::Vector3i min = -(shape/2);
    Eigen::Vector3i max = shape/2 - Eigen::Vector3i::Ones();

    for(int x=min.x(); x<=max.y(); x++){
        for(int y=min.y(); y<=max.y(); y++){
            for(int z=min.z(); z<=max.z(); z++){
                size_t idx = (x + shape.x()/2) * shape.y() * shape.z() + (y + shape.y()/2) * shape.z() + (z + shape.z()/2);
                if(tags_cpu[idx] == INSIDE){
                    inside_voxels.emplace_back(Eigen::Vector3i(x, y, z));
                }
            }
        }
    }

    return inside_voxels;
}

std::vector<Eigen::Vector3i> DualVoxelDynamic::outside_voxel(Eigen::Vector3i shape){
    std::vector<Eigen::Vector3i> outside_voxels;
    size_t voxel_num = shape.x() * shape.y() * shape.z();
    GPUVector<Eigen::Vector3f> voxel_centers(voxel_num);
    GPUVector<float> distances(voxel_num);
    GPUVector<char> tags_gpu(voxel_num);
    std::vector<char> tags_cpu(voxel_num);

    const dim3 threads = {16, 16, 2};
    const dim3 blocks = {div_round_up((uint32_t)shape.x(), 16u), div_round_up((uint32_t)shape.y(), 16u), div_round_up((uint32_t)shape.z(), 2u)};

    uniform_sample_voxel_center<<<blocks, threads, 0, stream>>>( voxel_centers.ptr(), shape);
    bvh->signed_distance_gpu(voxel_centers.ptr(), distances.ptr(), voxel_num, triangles_gpu.ptr(), m_mode, false, stream);

    classify_voxels<<<div_round_up(voxel_num, (size_t)128), 128, 0, stream>>>(distances.ptr(), tags_gpu.ptr(), voxel_num, voxel_size(shape).minCoeff()/2);

    tags_gpu.memcpyto(&tags_cpu[0], voxel_num, stream);

    
    Eigen::Vector3i min = -(shape/2);
    Eigen::Vector3i max = shape/2 - Eigen::Vector3i::Ones();

    for(int x=min.x(); x<=max.y(); x++){
        for(int y=min.y(); y<=max.y(); y++){
            for(int z=min.z(); z<=max.z(); z++){
                size_t idx = (x + shape.x()/2) * shape.y() * shape.z() + (y + shape.y()/2) * shape.z() + (z + shape.z()/2);
                if(tags_cpu[idx] == OUTSIDE){
                    outside_voxels.emplace_back(Eigen::Vector3i(x, y, z));
                }
            }
        }
    }

    return outside_voxels;
}

std::vector<Eigen::Vector3f> DualVoxelDynamic::surface_voxel_center(Eigen::Vector3i shape){
    std::vector<Eigen::Vector3f> surface_voxel_centers;

    size_t voxel_num = shape.x() * shape.y() * shape.z();
    GPUVector<Eigen::Vector3f> voxel_centers(voxel_num);
    GPUVector<float> distances(voxel_num);
    GPUVector<char> tags_gpu(voxel_num);
    std::vector<char> tags_cpu(voxel_num);

    const dim3 threads = {16, 16, 2};
    const dim3 blocks = {div_round_up((uint32_t)shape.x(), 16u), div_round_up((uint32_t)shape.y(), 16u), div_round_up((uint32_t)shape.z(), 2u)};

    uniform_sample_voxel_center<<<blocks, threads, 0, stream>>>( voxel_centers.ptr(), shape);
    bvh->signed_distance_gpu(voxel_centers.ptr(), distances.ptr(), voxel_num, triangles_gpu.ptr(), m_mode, false, stream);

    classify_voxels<<<div_round_up(voxel_num, (size_t)128), 128, 0, stream>>>(distances.ptr(), tags_gpu.ptr(), voxel_num, voxel_size(shape).minCoeff()/2);
    tags_gpu.memcpyto(&tags_cpu[0], voxel_num, stream);

    for(int x=0; x<shape.x(); x++){
        for(int y=0; y<shape.y(); y++){
            for(int z=0; z<shape.z(); z++){
                size_t idx = x * shape.y() * shape.z() + y * shape.z() + z;
                if(tags_cpu[idx] == SURFACE){
                    surface_voxel_centers.emplace_back(thread_to_center({x, y, z}, shape));
                }
            }
        }
    }

    return surface_voxel_centers;
}

std::vector<Eigen::Vector3f> DualVoxelDynamic::inside_voxel_center(Eigen::Vector3i shape){
    std::vector<Eigen::Vector3f> inside_voxel_centers;
    size_t voxel_num = shape.x() * shape.y() * shape.z();
    GPUVector<Eigen::Vector3f> voxel_centers(voxel_num);
    GPUVector<float> distances(voxel_num);
    GPUVector<char> tags_gpu(voxel_num);
    std::vector<char> tags_cpu(voxel_num);

    const dim3 threads = {16, 16, 2};
    const dim3 blocks = {div_round_up((uint32_t)shape.x(), 16u), div_round_up((uint32_t)shape.y(), 16u), div_round_up((uint32_t)shape.z(), 2u)};

    uniform_sample_voxel_center<<<blocks, threads, 0, stream>>>( voxel_centers.ptr(), shape);
    bvh->signed_distance_gpu(voxel_centers.ptr(), distances.ptr(), voxel_num, triangles_gpu.ptr(), m_mode, false, stream);

    classify_voxels<<<div_round_up(voxel_num, (size_t)128), 128, 0, stream>>>(distances.ptr(), tags_gpu.ptr(), voxel_num, voxel_size(shape).minCoeff()/2);

    tags_gpu.memcpyto(&tags_cpu[0], voxel_num, stream);

    Eigen::Vector3i min = -(shape/2);
    Eigen::Vector3i max = shape/2 - Eigen::Vector3i::Ones();

    for(int x=min.x(); x<=max.y(); x++){
        for(int y=min.y(); y<=max.y(); y++){
            for(int z=min.z(); z<=max.z(); z++){
                size_t idx = (x + shape.x()/2) * shape.y() * shape.z() + (y + shape.y()/2) * shape.z() + (z + shape.z()/2);
                if(tags_cpu[idx] == INSIDE){
                    inside_voxel_centers.emplace_back(voxel_center({x, y, z}, shape));
                }
            }
        }
    }

    return inside_voxel_centers;
}

std::vector<Eigen::Vector3f> DualVoxelDynamic::outside_voxel_center(Eigen::Vector3i shape){
    std::vector<Eigen::Vector3f> outside_voxel_centers;
    size_t voxel_num = shape.x() * shape.y() * shape.z();
    GPUVector<Eigen::Vector3f> voxel_centers(voxel_num);
    GPUVector<float> distances(voxel_num);
    GPUVector<char> tags_gpu(voxel_num);
    std::vector<char> tags_cpu(voxel_num);

    const dim3 threads = {16, 16, 2};
    const dim3 blocks = {div_round_up((uint32_t)shape.x(), 16u), div_round_up((uint32_t)shape.y(), 16u), div_round_up((uint32_t)shape.z(), 2u)};

    uniform_sample_voxel_center<<<blocks, threads, 0, stream>>>( voxel_centers.ptr(), shape);
    bvh->signed_distance_gpu(voxel_centers.ptr(), distances.ptr(), voxel_num, triangles_gpu.ptr(), m_mode, false, stream);

    classify_voxels<<<div_round_up(voxel_num, (size_t)128), 128, 0, stream>>>(distances.ptr(), tags_gpu.ptr(), voxel_num, voxel_size(shape).minCoeff()/2);

    tags_gpu.memcpyto(&tags_cpu[0], voxel_num, stream);

    Eigen::Vector3i min = -(shape/2);
    Eigen::Vector3i max = shape/2 - Eigen::Vector3i::Ones();

    for(int x=min.x(); x<=max.y(); x++){
        for(int y=min.y(); y<=max.y(); y++){
            for(int z=min.z(); z<=max.z(); z++){
                size_t idx = (x + shape.x()/2) * shape.y() * shape.z() + (y + shape.y()/2) * shape.z() + (z + shape.z()/2);
                if(tags_cpu[idx] == OUTSIDE){
                    outside_voxel_centers.emplace_back(voxel_center({x, y, z}, shape));
                }
            }
        }
    }

    return outside_voxel_centers;
}

std::vector<char> DualVoxelDynamic::judge_voxel(std::vector<Eigen::Vector3i> &voxels, Eigen::Vector3i shape){
    std::vector<Eigen::Vector3f> voxelsf(voxels.size());
    for(size_t i = 0; i < voxels.size(); i++){
        voxelsf[i] = voxel_center(voxels[i], shape);
    }
    float voxel_radius = (voxel_size(shape)/2).minCoeff();
    return judge_voxel_center(voxelsf, voxel_radius);
}

std::vector<char> DualVoxelDynamic::judge_voxel_center(std::vector<Eigen::Vector3f> &centers, float voxel_radius){
    GPUVector<Eigen::Vector3f> centers_gpu;
    centers_gpu.resize_and_memcpy_from_vector(centers, stream);
    GPUVector<float> distances(centers.size());
    GPUVector<char> tag_gpu(centers.size());
    std::vector<char> tag_cpu(centers.size());

    bvh->signed_distance_gpu(centers_gpu.ptr(), distances.ptr(), centers.size(), triangles_gpu.ptr(), m_mode, false, stream);

    classify_voxels<<<div_round_up(centers.size(), (size_t)128), 128, 0, stream>>>(distances.ptr(), tag_gpu.ptr(), centers.size(), voxel_radius);

    tag_gpu.memcpyto(&tag_cpu[0], centers.size(), stream);

    return tag_cpu;
}

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Dual Voxel<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

size_t DualVoxel::voxelize_triangles(std::vector<Triangle> &triangles){

    float voxel_radius = voxel_size().minCoeff()/2;
    m_surface_num = 0; 
    
    for(auto t : triangles){
        // 1. define a local area to reduce calculate cost.
        BoundingBox aabb;
        aabb.update(t);

        // 2. determine voxels in integer system
        Eigen::Vector3i maxi = point_in_voxel(aabb.max);
        Eigen::Vector3i mini = point_in_voxel(aabb.min);

        // 4. traverse all voxels
        for(int x = mini.x(); x <= maxi.x(); x++){
            for(int y = mini.y(); y <= maxi.y(); y++){
                for(int z = mini.z(); z <= maxi.z(); z++){
                    uint32_t idx = hash({x, y, z});
                    if(m_grid[idx] != 1){    // skip voxels occupied
                        Eigen::Vector3f voxel_center = center({x, y, z});
                        if(triangle_test(voxel_center, t.a, t.b, t.c, voxel_radius)){
                            m_grid[idx] = SURFACE;
                            m_surface_num++;
                        }
                    }
                }
            }
        }
    }
    return m_surface_num;
}

size_t DualVoxel::fill_inside(){
    if(!m_surface_num)return 0;
    std::queue<Eigen::Vector3i> Q;

    // we set this voxel as seed because it work stable while all model 
    // is watertight
    Eigen::Vector3i seed = {-(m_shape.x()/2), -(m_shape.y()/2), -(m_shape.z()/2)};
    // the point should be set outside before push
    // otherwise this voxel will be calculate one more times
    size_t idx = hash(seed);
    m_grid[idx] = OUTSIDE;
    m_inside_num = total_num() - surface_num() - 1;

    Q.push(seed);
    size_t idx_adjcent;

    // 4. BFS
    while(!Q.empty()){
        Eigen::Vector3i p = Q.front();
        // search adjcent voxels
        // 6 adject voxels
        if(p.x() > -(m_shape.x()/2)){
            idx_adjcent = hash({p.x()-1, p.y(), p.z()});
            if(m_grid[idx_adjcent] != SURFACE && m_grid[idx_adjcent] != OUTSIDE){
                m_grid[idx_adjcent] = OUTSIDE;
                m_inside_num--;
                Q.push({p.x()-1, p.y(), p.z()});
            } 
        }
        if(p.x() <  (m_shape.x()/2-1)){
            idx_adjcent = hash({p.x()+1, p.y(), p.z()});
            if(m_grid[idx_adjcent] != SURFACE && m_grid[idx_adjcent] != OUTSIDE){
                m_grid[idx_adjcent] = OUTSIDE;
                m_inside_num--;
                Q.push({p.x()+1, p.y(), p.z()});
            }
        }
        if(p.y() > -(m_shape.y()/2)){
            idx_adjcent = hash({p.x(), p.y()-1, p.z()});
            if(m_grid[idx_adjcent] != SURFACE && m_grid[idx_adjcent] != OUTSIDE){
                m_grid[idx_adjcent] = OUTSIDE;
                m_inside_num--;
                Q.push({p.x(), p.y()-1, p.z()});
            }
        }
        if(p.y() <  (m_shape.y()/2-1)){
            idx_adjcent = hash({p.x(), p.y()+1, p.z()});
            if(m_grid[idx_adjcent] != SURFACE && m_grid[idx_adjcent] != OUTSIDE){
                m_grid[idx_adjcent] = OUTSIDE;
                m_inside_num--;
                Q.push({p.x(), p.y()+1, p.z()});
            }  
        }
        if(p.z() > -(m_shape.z()/2)){
            idx_adjcent = hash({p.x(), p.y(), p.z()-1});
            if(m_grid[idx_adjcent] != SURFACE && m_grid[idx_adjcent] != OUTSIDE){
                m_grid[idx_adjcent] = OUTSIDE;
                m_inside_num--;
                Q.push({p.x(), p.y(), p.z()-1});
            } 
        }
        if(p.z() <  (m_shape.z()/2-1)){
            idx_adjcent = hash({p.x(), p.y(), p.z()+1});
            if(m_grid[idx_adjcent] != SURFACE && m_grid[idx_adjcent] != OUTSIDE){
                m_grid[idx_adjcent] = OUTSIDE;
                m_inside_num--;
                Q.push({p.x(), p.y(), p.z()+1});
            }
        }
        Q.pop();
    }
    return m_inside_num;
}

std::vector<Eigen::Vector3i> DualVoxel::surface_voxel(){
    if(!m_surface_num)return {};
    std::vector<Eigen::Vector3i> surfaces(m_surface_num);
    
    Eigen::Vector3i min = -(m_shape/2);
    Eigen::Vector3i max = (m_shape/2) - Eigen::Vector3i::Ones();

    size_t i = 0;
    for(int x = min.x(); x <= max.x(); x++){
        for(int y = min.y(); y <= max.y(); y++){
            for(int z = min.z(); z <= max.z(); z++){
                size_t idx = hash({x, y, z});
                if(m_grid[idx] == SURFACE){
                    surfaces[i++] = {x, y, z};
                }
            }
        }
    }
    return surfaces;
}

std::vector<Eigen::Vector3i> DualVoxel::inside_voxel(){
    if(!m_inside_num)return {};
    std::vector<Eigen::Vector3i> inners(m_inside_num);
    
    Eigen::Vector3i min = -(m_shape/2);
    Eigen::Vector3i max = (m_shape/2) - Eigen::Vector3i::Ones();

    size_t i = 0;
    for(int x = min.x(); x <= max.x(); x++){
        for(int y = min.y(); y <= max.y(); y++){
            for(int z = min.z(); z <= max.z(); z++){
                size_t idx = hash({x, y, z});
                if(m_grid[idx] == INSIDE){
                    inners[i++] = {x, y, z};
                }
            }
        }
    }
    return inners;
}

std::vector<Eigen::Vector3i> DualVoxel::outside_voxel(){
    if(!outside_num())return {};
    std::vector<Eigen::Vector3i> outers(outside_num());
    
    Eigen::Vector3i min = -(m_shape/2);
    Eigen::Vector3i max = (m_shape/2) - Eigen::Vector3i::Ones();

    size_t i = 0;
    for(int x = min.x(); x <= max.x(); x++){
        for(int y = min.y(); y <= max.y(); y++){
            for(int z = min.z(); z <= max.z(); z++){
                size_t idx = hash({x, y, z});
                if(m_grid[idx] == OUTSIDE){
                    outers[i++] = {x, y, z};
                }
            }
        }
    }
    return outers;
}

std::vector<Eigen::Vector3f> DualVoxel::surface_voxel_center(){
    if(!m_surface_num)return {};
    std::vector<Eigen::Vector3f> surfaces(m_surface_num);
    
    Eigen::Vector3i min = -(m_shape/2);
    Eigen::Vector3i max = (m_shape/2) - Eigen::Vector3i::Ones();

    size_t i = 0;
    for(int x = min.x(); x <= max.x(); x++){
        for(int y = min.y(); y <= max.y(); y++){
            for(int z = min.z(); z <= max.z(); z++){
                size_t idx = hash({x, y, z});
                if(m_grid[idx] == SURFACE){
                    surfaces[i++] = center({x, y, z});
                }
            }
        }
    }
    return surfaces;
}

std::vector<Eigen::Vector3f> DualVoxel::inside_voxel_center(){
    if(!m_inside_num)return {};
    std::vector<Eigen::Vector3f> inners(m_inside_num);
    
    Eigen::Vector3i min = -(m_shape/2);
    Eigen::Vector3i max = (m_shape/2) - Eigen::Vector3i::Ones();

    size_t i = 0;
    for(int x = min.x(); x <= max.x(); x++){
        for(int y = min.y(); y <= max.y(); y++){
            for(int z = min.z(); z <= max.z(); z++){
                size_t idx = hash({x, y, z});
                if(m_grid[idx] == INSIDE){
                    inners[i++] = center({x, y, z});
                }
            }
        }
    }
    return inners;
}

std::vector<Eigen::Vector3f> DualVoxel::outside_voxel_center(){
    if(!outside_num())return {};
    std::vector<Eigen::Vector3f> outers(outside_num());
    
    Eigen::Vector3i min = -(m_shape/2);
    Eigen::Vector3i max = (m_shape/2) - Eigen::Vector3i::Ones();

    size_t i = 0;
    for(int x = min.x(); x <= max.x(); x++){
        for(int y = min.y(); y <= max.y(); y++){
            for(int z = min.z(); z <= max.z(); z++){
                size_t idx = hash({x, y, z});
                if(m_grid[idx] == OUTSIDE){
                    outers[i++] = center({x, y, z});
                }
            }
        }
    }
    return outers;
}

std::vector<Eigen::Vector3f> DualVoxel::voxel_centers(){
    std::vector<Eigen::Vector3f> res(total_num());
    for(int x=min().x(); x <= max().x(); x++){
        for(int y=min().y(); y <= max().y(); y++){
            for(int z=min().z(); z <= max().z(); z++){
                size_t idx = hash({x, y, z});
                res[idx] = center({x, y, z});
            }
        }
    }
    return res;
}

std::vector<Eigen::Vector3i> DualVoxel::voxels(){
    std::vector<Eigen::Vector3i> res(total_num());
    for(int x=min().x(); x <= max().x(); x++){
        for(int y=min().y(); y <= max().y(); y++){
            for(int z=min().z(); z <= max().z(); z++){
                size_t idx = hash({x, y, z});
                res[idx] = Eigen::Vector3i(x, y, z);
            }
        }
    }
    return res;
}

size_t DualVoxel::HYD::hash(Eigen::Vector3i p, Eigen::Vector3i shape){
    Eigen::Vector3i n = p + shape/2;
    return (size_t)n.x() * shape.y() * shape.z() + n.y() * shape.z() + n.z();
}

bool DualVoxel::HYD::edge_check(std::vector<int8_t> &grid, Eigen::Vector3i shape, Eigen::Vector3i p){
    size_t idx = hash(p, shape);
    if(grid[idx]){
        size_t nidx;
        // left 9 
        nidx = hash({p.x()-1, p.y()-1, p.z()-1}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()-1, p.y()-1, p.z()}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()-1, p.y()-1, p.z()+1}, shape);
        if(!grid[nidx])return true;

        nidx = hash({p.x()-1, p.y(), p.z()-1}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()-1, p.y(), p.z()}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()-1, p.y(), p.z()+1}, shape);
        if(!grid[nidx])return true;

        nidx = hash({p.x()-1, p.y()+1, p.z()-1}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()-1, p.y()+1, p.z()}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()-1, p.y()+1, p.z()+1}, shape);
        if(!grid[nidx])return true;
        // mid 8
        nidx = hash({p.x(), p.y()-1, p.z()-1}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x(), p.y()-1, p.z()}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x(), p.y()-1, p.z()+1}, shape);
        if(!grid[nidx])return true;

        nidx = hash({p.x(), p.y(), p.z()-1}, shape);
        if(!grid[nidx])return true;

        nidx = hash({p.x(), p.y(), p.z()+1}, shape);
        if(!grid[nidx])return true;
        
        nidx = hash({p.x(), p.y()+1, p.z()-1}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x(), p.y()+1, p.z()}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x(), p.y()+1, p.z()+1}, shape);
        if(!grid[nidx])return true;
        // right 9
        nidx = hash({p.x()+1, p.y()-1, p.z()-1}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()+1, p.y()-1, p.z()}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()+1, p.y()-1, p.z()+1}, shape);
        if(!grid[nidx])return true;

        nidx = hash({p.x()+1, p.y(), p.z()-1}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()+1, p.y(), p.z()}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()+1, p.y(), p.z()+1}, shape);
        if(!grid[nidx])return true;
        
        nidx = hash({p.x()+1, p.y()+1, p.z()-1}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()+1, p.y()+1, p.z()}, shape);
        if(!grid[nidx])return true;
        nidx = hash({p.x()+1, p.y()+1, p.z()+1}, shape);
        if(!grid[nidx])return true;
    }
    return false;
}

std::vector<size_t> DualVoxel::HYD::edge_voxel(std::vector<int8_t> &grid, Eigen::Vector3i shape){
    std::vector<size_t> res;
    Eigen::Vector3i min = -(shape/2);
    Eigen::Vector3i max = shape/2 - Eigen::Vector3i::Ones();
    for(int x=min.x(); x <=max.x(); x++){
        for(int y=min.y(); y<=max.y(); y++){
            for(int z=min.z(); z<= max.z(); z++){
                if(edge_check(grid, shape, {x, y, z})){
                    res.emplace_back(hash({x, y, z}, shape));
                }
            }
        }
    }
    return res;
}

__global__ void subvoxel_centers_kenrel(
    Eigen::Vector3f *centers_low,
    Eigen::Vector3f *centers_high,
    uint32_t num,
    Eigen::Vector3i shape_low,
    Eigen::Vector3i shape_high
){
    const uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= num)return;

    auto rate = shape_high.cwiseQuotient(shape_low);
    size_t ratef = rate.x() * rate.y() * rate.z();
    auto center = centers_low[idx];

    Eigen::Vector3i minb = -(rate/2);
    Eigen::Vector3i maxb = (rate/2) - Eigen::Vector3i::Ones();

    Eigen::Vector3f voxel_size = Eigen::Vector3f(2.0f/shape_high.x(), 2.0f/shape_high.y(), 2.0f/shape_high.z());

    uint32_t high_base = idx * ratef;

    for(int x = minb.x(); x <= maxb.x(); x++){
        for(int y = minb.y(); y <= maxb.y(); y++){
            for(int z = minb.z(); z <= maxb.z(); z++){
                Eigen::Vector3i n = Eigen::Vector3i(x, y, z) + (rate/2);
                uint32_t idx_high = high_base + n.x() * rate.y() * rate.z() + n.y() * rate.z() + n.z();
                Eigen::Vector3f f = Eigen::Vector3f(0.5+x, 0.5+y, 0.5+z);
                // scale and translate
                centers_high[idx_high] = Eigen::Vector3f(f.array() * voxel_size.array()) + center;
            }
        }
    }
}

std::vector<Eigen::Vector3f> DualVoxel::HYD::subvoxel_centers(std::vector<Eigen::Vector3f> &centers, Eigen::Vector3i shape_low, Eigen::Vector3i shape_high){
    cudaStream_t stream;
    CUDA_CHECK_THROW(cudaStreamCreate(&stream));

    Eigen::Vector3i rate = shape_high.cwiseQuotient(shape_low);
    int ratef = rate.x() * rate.y() * rate.z();

    std::vector<Eigen::Vector3f> res(ratef * centers.size());
    GPUVector<Eigen::Vector3f> res_gpu(ratef * centers.size());
    GPUVector<Eigen::Vector3f> centers_gpu(centers.size());
    centers_gpu.memcpyfrom(centers.data(), centers.size(), stream);

    subvoxel_centers_kenrel<<<div_round_up(centers.size(), (size_t)128), 128, 0, stream>>>(
        centers_gpu.ptr(),
        res_gpu.ptr(),
        centers.size(),
        shape_low,
        shape_high
    );

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    res_gpu.memcpyto(res.data(), res.size(), stream);
    return res;
}