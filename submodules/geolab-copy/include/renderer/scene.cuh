#pragma once
#ifndef SCENE
#define SCENE

#include<Eigen/Eigen>
#include<geometry/boundingbox.cuh>

struct BRDFParams {
	float metallic=0.f;
	float subsurface=0.f;
	float specular=1.f;
	float roughness=0.5f;
	float sheen=0.f;
	float clearcoat=0.f;
	float clearcoat_gloss=0.f;
};

struct Light{
    Eigen::Vector3f pos = Eigen::Vector3f(4.0f, 4.0f, 4.0f);
    Eigen::Vector3f ambient_color = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
    Eigen::Array4f background_color = Eigen::Array4f(1.0f, 1.0f, 1.0f, 1.0f);
    Eigen::Vector3f light_color = Eigen::Vector3f(0.8f, 0.8f, 0.8f);
	float kd = 0.8f;
	float specular = 1.0f;
	bool parallel = false;	// Is Parallel light else Point light
};

struct Scene{
    Eigen::Vector3f surface_color = Eigen::Vector3f::Ones();
	Eigen::Vector3f sky_color = Eigen::Vector3f(195.0f/255.0f, 215.0f/255.0f, 255.0f/255.0f);
    float slice_plane_z = 0.0f;
    float floor_y = -1.0f;
	BoundingBox aabb;
	float aabb_offset = 1e-3f;
};

#endif