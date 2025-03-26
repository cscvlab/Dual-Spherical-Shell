#pragma once
#ifndef SPHERE_SAMPLER
#define SPHERE_SAMPLER

#include<Eigen/Eigen>
#include<utils/math_ex.cuh>

inline void sampler_fibonacci(Eigen::Vector3f *pos, unsigned int n){
    for(int i=0; i<n; i++){
        float phi = std::acos(1 - 2*(i + 0.5)/n);
        float theta = 2.0f * 3.1415926 * i / GOLDEN_RATIO;
        pos[i].x() = std::cos(theta) * std::sin(phi);
        pos[i].y() = std::sin(theta) * std::sin(phi);
        pos[i].z() = std::cos(phi);
    }
}

inline std::vector<Eigen::Vector3f> sampler_fibonacci(unsigned int n){
    std::vector<Eigen::Vector3f> ret(n);
    sampler_fibonacci(&ret[0], n);
    return ret;
}

#endif