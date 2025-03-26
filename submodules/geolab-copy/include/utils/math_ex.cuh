#pragma once
#ifndef MATHEX
#define MATHEX

#include<Eigen/Eigen>

static constexpr float GOLDEN_RATIO = 1.6180339887498948482045868343656f;
static constexpr float PI = 3.14159265358979323846f;

__host__ __device__ inline float sign(float x){
	return copysignf(1.0f, x);
}

__host__ __device__ inline float square(float x){
	return x * x;
}

__host__ __device__ inline float clamp(float x, float min, float max){
	if(x<min)return min;
	else if(x > max)return max;
	else return x;
}

__host__ __device__ inline Eigen::Vector3f clamp(const Eigen::Vector3f &x, float min, float max){
	return Eigen::Vector3f(
		clamp(x.x(), min, max), 
		clamp(x.y(), min, max), 
		clamp(x.z(), min, max)
		);
}

__host__ __device__ inline float fractf(float x){
	return x - floorf(x);
}

template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float3 to_float3(Eigen::Vector3f p){
	return {p.x(), p.y(), p.z()};
}

inline __host__ __device__ Eigen::Vector3f faceforward(const Eigen::Vector3f& n, const Eigen::Vector3f& i, const Eigen::Vector3f& nref) {
	return n * copysignf(1.0f, i.dot(nref));
}

inline __host__ __device__ __device__ Eigen::Vector3f cosine_hemisphere(const Eigen::Vector2f& u) {
	// Uniformly sample disk
	const float r   = sqrtf(u.x());
	const float phi = 2.0f * PI * u.y();

	Eigen::Vector3f p;
	p.x() = r * cosf(phi);
	p.y() = r * sinf(phi);

	// Project up to hemisphere
	p.z() = sqrtf(fmaxf(0.0f, 1.0f - p.x()*p.x() - p.y()*p.y()));

	return p;
}

__host__ __device__ inline Eigen::Vector3f cylindrical_to_dir(const Eigen::Vector2f& p) {
	const float cos_theta = -2.0f * p.x() + 1.0f;
	const float phi = 2.0f * PI * (p.y() - 0.5f);

	const float sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));
	float sin_phi, cos_phi;
	sincosf(phi, &sin_phi, &cos_phi);

	return {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
}

// Reference Code from instant-ngp
template<unsigned int N_DIRS>
__host__ __device__ inline Eigen::Vector3f fibonacci_dir(unsigned int i, Eigen::Vector2f offset = Eigen::Vector2f::Zero()){
	float epsilon;
	if (N_DIRS >= 11000) {
		epsilon = 27;
	} else if (N_DIRS >= 890) {
		epsilon = 10;
	} else if (N_DIRS >= 177) {
		epsilon = 3.33;
	} else if (N_DIRS >= 24) {
		epsilon = 1.33;
	} else {
		epsilon = 0.33;
	}

	return cylindrical_to_dir(
        Eigen::Vector2f{
            fractf((i+epsilon) / (N_DIRS-1+2*epsilon) + offset.x()), 
			fractf(i / GOLDEN_RATIO + offset.y())
            }
        );
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ inline uint32_t expand_bits(uint32_t v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__host__ __device__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
	uint32_t xx = expand_bits(x);
	uint32_t yy = expand_bits(y);
	uint32_t zz = expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

__host__ __device__ inline uint32_t morton3D_invert(uint32_t x) {
	x = x               & 0x49249249;
	x = (x | (x >> 2))  & 0xc30c30c3;
	x = (x | (x >> 4))  & 0x0f00f00f;
	x = (x | (x >> 8))  & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

__host__ __device__ inline uint64_t expand_bits(uint64_t w)  {
	w &=                0x00000000001fffff;
	w = (w | w << 32) & 0x001f00000000ffff;
	w = (w | w << 16) & 0x001f0000ff0000ff;
	w = (w | w <<  8) & 0x010f00f00f00f00f;
	w = (w | w <<  4) & 0x10c30c30c30c30c3;
	w = (w | w <<  2) & 0x1249249249249249;
	return w;
}

__host__ __device__ inline uint64_t morton3D_64bit(uint32_t x, uint32_t y, uint32_t z)  {
	return ((expand_bits((uint64_t)x)) | (expand_bits((uint64_t)y) << 1) | (expand_bits((uint64_t)z) << 2));
}

inline __host__ __device__ uint32_t binary_search(float val, const float* data, uint32_t length) {
	if (length == 0) {
		return 0;
	}

	uint32_t it;
	uint32_t count, step;
	count = length;

	uint32_t first = 0;
	while (count > 0) {
		it = first;
		step = count / 2;
		it += step;
		if (data[it] < val) {
			first = ++it;
			count -= step + 1;
		} else {
			count = step;
		}
	}

	return std::min(first, length-1);
}

inline __host__ __device__ float logit(const float x) {	// 1e-9 < x < 1-1e-9
	// -log(1/x) - 1
	return -logf(1.0f / (fminf(fmaxf(x, 1e-9f), 1.0f - 1e-9f)) - 1.0f);
}

#endif