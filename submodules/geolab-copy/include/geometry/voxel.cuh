#pragma once
#ifndef VOXEL
#define VOXEL

#include<vector>
#include<memory>

#include<geometry/mesh.h>
#include<geometry/trianglebvh.cuh>
#include<sampler/space_sampler.cuh>

#define SURFACE 1
#define OUTSIDE -1
#define INSIDE 0

// This class use sdf to judge where is the voxel
// |sdf| < cell_size -> surface voxel
// |sdf| > cell_size && sdf > 0 -> outside voxel
// |sdf| > cell_size && sdf < 0 -> inner voxel
class DualVoxelDynamic{
    private:
        SDFCalcMode m_mode;

        std::unique_ptr<TriangleBVH> bvh;
        std::vector<Triangle> triangles_cpu;
        GPUVector<Triangle> triangles_gpu;
        cudaStream_t stream;
    public:
        DualVoxelDynamic(SDFCalcMode mode=SDFCalcMode::RAYSTAB): m_mode(mode){ CUDA_CHECK_THROW(cudaStreamCreate(&stream));}
        ~DualVoxelDynamic(){}

        void load_model(std::string obj_path);
        void load_model(std::vector<Triangle> &trianlges);
        
        std::vector<Eigen::Vector3i> surface_voxel(Eigen::Vector3i shape);
        std::vector<Eigen::Vector3i> inside_voxel(Eigen::Vector3i shape);
        std::vector<Eigen::Vector3i> outside_voxel(Eigen::Vector3i shape);

        std::vector<Eigen::Vector3f> surface_voxel_center(Eigen::Vector3i shape);
        std::vector<Eigen::Vector3f> inside_voxel_center(Eigen::Vector3i shape);
        std::vector<Eigen::Vector3f> outside_voxel_center(Eigen::Vector3i shape);

        std::vector<char> judge_voxel(std::vector<Eigen::Vector3i> &voxels, Eigen::Vector3i shape);
        std::vector<char> judge_voxel_center(std::vector<Eigen::Vector3f> &voxels, float voxel_radius);

};


// use flood to fill inside
class DualVoxel{
    public:
        DualVoxel(Eigen::Vector3i shape): m_grid(shape.x() * shape.y() * shape.z()), m_shape(shape){}
        ~DualVoxel(){}

        size_t surface_num() const {return m_surface_num;}
        size_t inside_num() const {return m_inside_num;}
        size_t voxel_num() const {return m_surface_num + m_inside_num;}
        size_t total_num() const {return m_shape.x() * m_shape.y() * m_shape.z();}
        size_t outside_num() const {return total_num() - voxel_num();}
        Eigen::Vector3i min() const {return -(m_shape/2);}
        Eigen::Vector3i max() const {return m_shape / 2 - Eigen::Vector3i::Ones();}

        Eigen::Vector3f voxel_size() const {return Eigen::Vector3f(2.0f/m_shape.x(), 2.0f/m_shape.y(), 2.0f/m_shape.z());}
        Eigen::Vector3f half_voxel_size() const {return Eigen::Vector3f(1.0f/m_shape.x(), 1.0f/m_shape.y(), 1.0f/m_shape.z());}
        Eigen::Vector3f center(Eigen::Vector3i p) const {
            Eigen::Array3f f = Eigen::Array3f(p.x()+0.5f, p.y()+0.5f, p.z()+0.5f);
            return f.array() * voxel_size().array();
        }
        Eigen::Vector3i point_in_voxel(Eigen::Vector3f p) const{
            Eigen::Vector3f f = p.cwiseQuotient(voxel_size());
            return Eigen::Vector3i((int)std::floor(f.x()), (int)std::floor(f.y()), (int)std::floor(f.z()));
        }
        size_t hash(Eigen::Vector3i p) const{
            Eigen::Vector3i n = p + m_shape/2;
            return (size_t)n.x() * m_shape.y() * m_shape.z() + n.y() * m_shape.z() + n.z();
        }

        size_t voxelize_triangles(std::vector<Triangle> &triangles);
        size_t fill_inside();

        bool save_grid(std::string path);
        bool load_grid(std::string path);

        std::vector<Eigen::Vector3i> surface_voxel();
        std::vector<Eigen::Vector3i> inside_voxel();
        std::vector<Eigen::Vector3i> outside_voxel();

        std::vector<Eigen::Vector3f> surface_voxel_center();
        std::vector<Eigen::Vector3f> inside_voxel_center();
        std::vector<Eigen::Vector3f> outside_voxel_center();

        std::vector<Eigen::Vector3i> voxels();
        std::vector<Eigen::Vector3f> voxel_centers();
        std::vector<char> tags(){return m_grid;}

        class HYD{
            public:
                static size_t hash(Eigen::Vector3i p, Eigen::Vector3i shape);
                static bool edge_check(std::vector<int8_t> &grid, Eigen::Vector3i shape, Eigen::Vector3i p);
                static std::vector<size_t> edge_voxel(std::vector<int8_t> &grid, Eigen::Vector3i shape);
                static std::vector<Eigen::Vector3f> subvoxel_centers(std::vector<Eigen::Vector3f> &centers, Eigen::Vector3i shape_low, Eigen::Vector3i shape_high);
        };
        
    private:
        std::vector<char> m_grid;
        Eigen::Vector3i m_shape;
        size_t m_surface_num = 0;
        size_t m_inside_num = 0;
};



#endif