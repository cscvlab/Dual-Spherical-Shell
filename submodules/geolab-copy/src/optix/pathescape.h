#pragma once
#include<utils/random.cuh>
#include <geometry/triangle.cuh>
#include <optix.h>

struct PathEscape {
	struct Params
	{
		const Eigen::Vector3f* ray_origins;
		const Triangle* triangles;
		float* distances;
		OptixTraversableHandle handle;
	};

	struct RayGenData {};
	struct MissData {};
	struct HitGroupData {};
};


