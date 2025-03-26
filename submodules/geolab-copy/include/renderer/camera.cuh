#pragma once
#ifndef CAMERA
#define CAMERA

#include<vector>
#include<math.h>
#include<Eigen/Eigen>
#include<utils/common.cuh>

struct Camera{
    Eigen::Matrix<float, 3, 4> pose;
    Eigen::Vector2f fov = {30.0f, 30.0f};
    float dof = 0.0f;
    float zoom = 1.0f;

    __host__ __device__ Camera(){
        pose.col(3) = Eigen::Vector3f(0.0f, 0.0f, 4.0f);
        focus(Eigen::Vector3f::Zero());
    }

    __host__ __device__ Camera(Eigen::Vector3f pos, Eigen::Vector3f focus_point = Eigen::Vector3f::Zero()){
        pose.col(3) = pos;
        focus(focus_point);
    }

    __host__ __device__ Eigen::Vector3f view_pos() const {return pose.col(3);}
    __host__ __device__ Eigen::Vector3f view_gaze() const {return pose.col(2);}
    __host__ __device__ Eigen::Vector3f view_up() const {return pose.col(1);}
    __host__ __device__ Eigen::Vector3f view_side() const {return pose.col(0);}

    __host__ __device__ void focus(Eigen::Vector3f focus_point = Eigen::Vector3f::Zero()){
        pose.col(2) = (focus_point - pose.col(3)).normalized();
        pose.col(1) = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        pose.col(0) = pose.col(2).cross(pose.col(1)).normalized();
        pose.col(1) = pose.col(0).cross(pose.col(2)).normalized();
    }

    __host__ __device__ Eigen::Vector2f calc_focal_length(Eigen::Vector2i win_res) const{
        return Eigen::Vector2f::Constant((0.5f * 1.0f / tanf(0.5f * fov[0] * 3.14159265f/180))) * win_res[0] * zoom;
    }

};

__host__ __device__ inline Eigen::Matrix4f matrix_view(const Eigen::Matrix<float, 3, 4> &pose) {
        Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f translate = Eigen::Matrix4f::Identity();
        auto pos = pose.col(3);
        auto gaze = pose.col(2);
        auto up = pose.col(1);
        auto side = pose.col(0);
        
        // translation
        translate.col(3) = Eigen::Vector4f(-pos.x(), -pos.y(), -pos.z(), 1.0f);
        matrix = translate * matrix;
        // rotation
        translate << side.x() , side.y() , side.z() , 0
                , up.x() , up.y() , up.z() , 0
                , gaze.x() , gaze.y() , gaze.z() , 0
                , 0.0f , 0.0f , 0.0f , 1.0f;
        matrix = translate * matrix;
        return matrix;
}

__host__ __device__ inline Eigen::Matrix4f matrix_proj(float zNear, float zFar, float eye_fov, float aspect_ratio){
        Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f translate;   // perspective transformation

        // perspective
        translate << zNear,    0,          0,           0,
                        0, zNear,          0,           0,
                        0,    0,    zNear * zFar,    zNear*zFar,
                        0,    0,          1,           0;
        matrix = translate * matrix;
        float t = zNear * tan(eye_fov/360*3.1415926);
        float r = aspect_ratio * t;
        translate << 1/r,   0,              0,               0,
                       0, 1/t,              0,               0,
                       0,   0, 2/(zNear - zFar)/zFar,        0,
                       0,   0,              0,               1;
        matrix = translate * matrix;
        
        return matrix;
}

__host__ __device__ inline Eigen::Matrix4f mvp(const Eigen::Matrix<float, 3, 4> &pose, float zNear, float zFar, float eye_fov, float aspect_ratio){
    return matrix_proj(zNear, zFar, eye_fov, aspect_ratio) * matrix_view(pose);
}

#endif