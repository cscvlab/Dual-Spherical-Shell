#include "ray_trace.h"

extern "C"{
    __constant__ RayTrace::Params params;
}

extern "C" __global__ void __raygen__rg(){
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    Eigen::Vector3f origin = params.origins[idx.x];
    Eigen::Vector3f direction = params.directions[idx.x];

    unsigned int p0, p1;    // hit, t
    // miss & ch below use two payload
    optixTrace(
        params.handle,
        to_float3(origin),
        to_float3(direction),
        0.0f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,
        1,
        0,
        p0, p1
    );

    float t = __int_as_float(p1);
    params.origins[idx.x] = origin + t * direction;

    if((int)p0 == -1)return;    // hit if p0 > 0
    Eigen::Vector3f normal = params.triangles[p0].normal();
    params.directions[idx.x] = normal.dot(direction) < 0 ? normal : (-normal);
}

extern "C" __global__ void __miss__ms(){
    optixSetPayload_0((uint32_t)-1);
    optixSetPayload_1(__float_as_int(optixGetRayTmax()));
}

// use two payload to record ray intersection
extern "C" __global__ void __closesthit__ch(){
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_int(optixGetRayTmax()));
}