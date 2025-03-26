#pragma once

#include<optix.h>

#include<geometry/triangle.cuh>

struct RayTrace{
    struct Params{
        Eigen::Vector3f *origins;
        Eigen::Vector3f *directions;
        const Triangle *triangles;
        OptixTraversableHandle handle;
    };

    struct RayGenData{};
    struct MissData{};
    struct HitGroupData{};
};
