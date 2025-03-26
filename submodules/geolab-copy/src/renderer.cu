#define STB_IMAGE_IMPLEMENTATION
#include<renderer/renderer.cuh>

inline __host__ __device__ Eigen::Vector2f ld_random_pixel_offset(const uint32_t /*x*/, const uint32_t /*y*/) {
    Eigen::Vector2f offset = Eigen::Vector2f::Constant(0.5f);
    offset.x() = fractf(offset.x());
    offset.y() = fractf(offset.y());
    return offset;
}

inline __host__ __device__ Eigen::Vector3f init_rays_direction(
    const Eigen::Vector2i& pixel,
    const Eigen::Vector2i& resolution,
    const Eigen::Vector2f& focal_length,
    const Eigen::Matrix<float, 3, 4>& camera_matrix,
    const Eigen::Vector2f& screen_center
) {
    Eigen::Vector2f offset = ld_random_pixel_offset(pixel.x(), pixel.y());
    Eigen::Vector2f uv = (pixel.cast<float>() + offset).cwiseQuotient(resolution.cast<float>());
    Eigen::Vector3f dir;
    
    dir = {
        (uv.x() - screen_center.x()) * (float)resolution.x() / focal_length.x(),
        (uv.y() - screen_center.y()) * (float)resolution.y() / focal_length.y(),
        1.0f
    };
    dir = camera_matrix.block<3, 3>(0, 0) * dir;
    return dir;
}


// return (1 - u)^5
__device__ inline float effect_SchlickFresnel(float u){
    float m = __saturatef(1.0f - u);
    return square(square(m)) * m;   
}

__device__ inline float G1(float NdotH, float a){
    if(a >= 1.0f)return 1.0f/PI;
    float a2 = square(a);
    float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
    return a2 / (PI * t * t);
}

//   a^2
// --------
// pi * (1 + (a^2 - 1) * (NdotH)^2)
__device__ inline float G2(float NdotH, float a){
    float a2 = square(a);
    float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
    return a2 / (PI * t * t);
}

__device__ inline float SmithG_GGX(float NdotV, float alphaG) {
	float a = alphaG * alphaG;
	float b = NdotV * NdotV;
	return 1.0 / (NdotV + sqrtf(a + b - a * b));
}

// return (1 - t) * a + t * b
__device__ inline float mix(float a, float b, float t){
    return a + (b - a) * t;
}

__device__ inline Eigen::Vector3f mix(Eigen::Vector3f a, Eigen::Vector3f b, float t){
    return a + (b - a) * t;
}

__device__ Eigen::Vector3f evaluate_shading(
    Eigen::Vector3f surface_color,
    Eigen::Vector3f ambient_color,
    Eigen::Vector3f light_color,
    float metallic,
    float subsurface,
    float specular,
    float roughness,
    float specular_tint,
    float sheen,
    float sheen_tint,
    float clearcoat,
    float clearcoat_gloss,
    Eigen::Vector3f L,  // light
    Eigen::Vector3f V,  // view
    Eigen::Vector3f N   // normal
){
    float NdotL = N.dot(L);
    float NdotV = N.dot(V);

    Eigen::Vector3f H = (L + V).normalized();
    float NdotH = N.dot(H);
    float LdotH = L.dot(H);

    // Diffuse fresnel 
    float FL = effect_SchlickFresnel(NdotL), FV = effect_SchlickFresnel(NdotV);
    Eigen::Vector3f amb = (ambient_color * mix(0.2f, FV, metallic));
    amb = amb.array() * surface_color.array();
    if(NdotL < 0.0f || NdotV < 0.0f)return amb; // if look from

    float luminance = surface_color.dot(Eigen::Vector3f(0.3f, 0.6f, 0.1f));

    // normalize luminance to isolate and saturation components
    Eigen::Vector3f Ctint = surface_color * (1.0f / (luminance+1e-5f));
    Eigen::Vector3f Cspec0 = mix(mix(Eigen::Vector3f::Ones(), Ctint, specular_tint) * specular * 0.08f, surface_color, metallic);
    Eigen::Vector3f Csheen = mix(Eigen::Vector3f::Ones(), Ctint, sheen_tint);

    float Fd90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
    float Fd = mix(1, Fd90, FL) * mix(1.0f, Fd90, FV);

    // Based on Hanrahan-Krueger BRDF approximation of isotropic BSSRDF
	// 1.25 scale is used to (roughly) preserve albedo
	// Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.0f / (NdotL + NdotV) - 0.5f) + 0.5f);

    float a = std::max(1e-3f, square(roughness));
    float Ds = G2(NdotH, a);
    float FH = effect_SchlickFresnel(LdotH);
    Eigen::Vector3f Fs = mix(Cspec0, Eigen::Vector3f::Ones(), FH);
    float Gs = SmithG_GGX(NdotL, a) * SmithG_GGX(NdotV, a);

    Eigen::Vector3f Fsheen = FH * sheen * Csheen;
    
    float Dr = G1(NdotH, mix(0.1f, 1e-3f, clearcoat_gloss));
    float Fr = mix(0.04f, 1.0f, FH);
    float Gr = SmithG_GGX(NdotL, 0.25f) * SmithG_GGX(NdotV, 0.25f);

    float CCs = 0.25f * clearcoat * Gr * Fr * Dr;
    Eigen::Vector3f brdf = (float(1.0f/PI) * mix(Fd, ss, subsurface) * surface_color + Fsheen) * (1.0f - metallic) + Gs * Fs * Ds + Eigen::Vector3f::Constant(CCs);
    return Eigen::Vector3f(brdf.array() * light_color.array()) * NdotL + amb;
}

__global__ void init_rays_from_camera_kernel(Eigen::Vector3f *positions,
                                             float *distances,
                                             SDFPayload *payloads,
                                             Eigen::Vector2i resolution,
                                             Eigen::Vector2f focal_length,
                                             Eigen::Matrix<float, 3, 4> camera_matrix,
                                             Eigen::Vector2f screen_center,
                                             BoundingBox aabb,
                                             float floor_y,
                                             float slice_plane_z){  // slice view
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= resolution.x() || y >= resolution.y()) return;

    uint32_t idx = x + resolution.x() * y;

    Eigen::Vector3f origin = camera_matrix.col(3);
    Eigen::Vector3f direction = init_rays_direction({x, y}, resolution, focal_length, camera_matrix, screen_center);
    direction = direction.normalized();

    if(slice_plane_z < 0){
        SDFPayload &payload = payloads[idx];
        payload.dir = direction;
        payload.idx = idx;
        payload.alive = false;
        payload.n_steps = 0;
        positions[idx] = origin - slice_plane_z * direction;
        distances[idx] = 10000.0f; 
        return;
    }

    float t = std::max(aabb.ray_intersect(origin, direction).x(), 0.0f);
    origin = origin + (t + 1e-6f) * direction;
    positions[idx] = origin;

    SDFPayload &payload = payloads[idx];
    if(!aabb.contains(origin)){
        distances[idx] = 10000.0f;
        payload.alive = false;
        return;
    }

    distances[idx] = 10000.0f;
    payload.dir = direction;
    payload.idx = idx;
    payload.n_steps = 0;
    payload.alive = true;
}

__global__ void init_rays_from_positions_and_directions_kernel(
    Eigen::Vector3f *directions,
    SDFPayload *payloads,
    uint32_t num
){
    const uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= num)return;
    payloads[idx].dir = directions[idx];
}

__global__ void payload_direction_abuse_normal(Eigen::Vector3f *normals, SDFPayload *payloads, size_t num){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num)return;
    normals[idx] = payloads[idx].dir;
}

__global__ void compact_rays_kernel(
    Eigen::Vector3f *src_positions, float *src_distances, SDFPayload *src_payloads,
    Eigen::Vector3f *dst_positions, float *dst_distances, SDFPayload *dst_payloads,
    Eigen::Vector3f *dst_final_positions, float *dst_final_distances, SDFPayload *dst_final_payloads,
    uint32_t num,
    BoundingBox aabb,
    uint32_t *counter, uint32_t *final_counter
){
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= num)return;
    SDFPayload &src_payload = src_payloads[i];
    if(src_payload.alive){
        uint32_t idx = atomicAdd(counter, 1);
        dst_payloads[idx] = src_payload;
        dst_positions[idx] = src_positions[i];
        dst_distances[idx] = src_distances[i];
    }else if(aabb.contains(src_positions[i])){
        uint32_t idx = atomicAdd(final_counter, 1);
        dst_final_payloads[idx] = src_payload;
        dst_final_positions[idx] = src_positions[i];
        dst_final_distances[idx] = src_distances[i];
    }
}

__global__ void compact_outer_rays_kernel(
    Eigen::Vector3f *src_positions, float *src_distances, SDFPayload *src_payloads,
    Eigen::Vector3f *dst_positions, float *dst_distances, SDFPayload *dst_payloads,
    uint32_t num, uint32_t *counter
){
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= num)return;
    SDFPayload &src_payload = src_payloads[i];
    if(src_distances > 0){
        uint32_t idx = atomicAdd(counter, 1);
        dst_payloads[idx] = src_payload;
        dst_positions[idx] = src_positions[i];
        dst_distances[idx] = src_distances[i];
    }
}

__global__ void sdf_advance_position_kernel(
    Eigen::Vector3f *positions,
    float *distances,
    SDFPayload *payloads,
    float *prev_distances,
    float *total_distances,
    uint32_t num,
    BoundingBox aabb,
    float distance_scale,
    float maximum_distance
){
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= num)return;

    SDFPayload &payload = payloads[i];
    float distance = distances[i];

    distance *= distance_scale;
    Eigen::Vector3f position = positions[i];
    position += distance * payload.dir;

    if(total_distances && distance > 0.0f){
        float total_distance = total_distances[i];
        float y = distance*distance / (2.0f * prev_distances[i]);
        float d = sqrtf(distance*distance - y*y);
        prev_distances[i] = distance;
        total_distances[i] = total_distance + distance;
    }

    bool stay_alive = distance > maximum_distance && fabsf(distance / 2) > 3*maximum_distance;

    if(!stay_alive){
        payload.alive = false;
        return;
    }

    if(!aabb.contains(position)){
        payload.alive = false;
        return;
    }

    payload.n_steps++;
}

__global__ void prepare_shadow_rays(
    Eigen::Vector3f *positions,
    Eigen::Vector3f *normals,
    float *distances,
    float *prev_distances,
    float *total_distances,
    float *min_visibility,
    SDFPayload *payloads,
    uint32_t num,
    BoundingBox aabb,
    Eigen::Vector3f light_pos,
    bool parallel
){
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= num)return;

    SDFPayload &payload = payloads[i];

    Eigen::Vector3f view_pos = positions[i] + faceforward(normals[i], -payload.dir, normals[i]) * 1e-3f;
    Eigen::Vector3f dir = parallel ? light_pos.normalized() : (light_pos - view_pos).normalized();

    // why first intersect with aabb ?
    float t = fmaxf(aabb.ray_intersect(view_pos, dir).x() + 1e-6f, 0.0f);
    view_pos += t * dir;

    positions[i] = view_pos;

    if(!aabb.contains(view_pos)){
        distances[i] = 10000.0f;
        payload.alive = false;
        min_visibility[i] = 1.0f;
        return;
    }

    distances[i] = 10000.0f;
    payload.idx = i;
    payload.dir = dir;
    payload.n_steps = 0;
    payload.alive = true;

    if (prev_distances) {
		prev_distances[i] = 1e20f;
	}

	if (total_distances) {
		total_distances[i] = 0.0f;
	}

	if (min_visibility) {
		min_visibility[i] = 1.0f;
	}
}

__global__ void write_shadow_ray_result(
    Eigen::Vector3f* positions, 
    SDFPayload* shadow_payloads, 
    float* min_visibility, 
    float* shadow_factors,
    uint32_t n_elements, 
    BoundingBox aabb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	shadow_factors[shadow_payloads[i].idx] = aabb.contains(positions[i]) ? 0.0f : min_visibility[i];
}

__global__ void shade_sdf_kernel(Eigen::Vector3f *positions,
                                 Eigen::Vector3f *normals,
                                 float *distances,
                                 SDFPayload *payloads,
                                 Eigen::Array4f *renderbuffer,
                                 const uint32_t num,
                                 BoundingBox aabb,
                                 ERenderMode render_mode,
                                 Eigen::Matrix<float, 3, 4> camera_matrix,
                                 Light light,
                                 BRDFParams brdf,
                                 Scene scene){
    const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= num)return;

    if(!aabb.contains(positions[i])){
        return;
    }
    
    SDFPayload &payload = payloads[i];
    Eigen::Vector3f normal = normals[i].normalized();
    Eigen::Vector3f position = positions[i];

    bool floor =  false;
    if(position.y() < scene.floor_y && payload.dir.y() < 0.0f){
        normal = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        floor = true;
    }

    Eigen::Vector3f camera_pos = camera_matrix.col(3);
    Eigen::Vector3f camera_gaze = camera_matrix.col(2);

    Eigen::Array3f color;
    switch(render_mode){
        case ERenderMode::LATTICE: {
            Eigen::Vector3f n = normal; 
            Eigen::Vector3f l = light.parallel ? light.pos.normalized() : (light.pos - position).normalized();
            Eigen::Vector3f v = -payload.dir.normalized();
            Eigen::Vector3f h = (v + l).normalized();
        
            float spec = std::abs(h.dot(n));
            spec = std::pow(spec, 16.0);

            Eigen::Vector3f diffuse_color = light.light_color.array() * Eigen::Vector3f::Constant(light.kd * std::abs(l.dot(n))).array();
            diffuse_color = clamp(light.ambient_color + diffuse_color, 0.0f, 1.0f);
            Eigen::Vector3f specular_color = clamp((light.light_color * light.specular * spec), 0.0f, 1.0f);
            
            color = clamp(Eigen::Vector3f(scene.surface_color.array() * diffuse_color.array()) + specular_color, 0.0f, 1.0f);
            
        } break;
        case ERenderMode::DIFFUSE: {
            Eigen::Vector3f l = light.parallel ? light.pos.normalized() : (light.pos - position).normalized();
            color = Eigen::Array3f(std::abs(normal.dot(l)));
        } break;
        case ERenderMode::Shade: {
            float skyam = normal.dot(Eigen::Vector3f(0.0f, 1.0f, 0.0f)) * -0.5f + 0.5f;
            Eigen::Vector3f sun_color = light.light_color * 4.0f * distances[i];
            const Eigen::Vector3f sky_color = scene.sky_color * 4.0f * skyam;
            float check_size = 8.0f / aabb.diag().x();
            float check=((int(floorf(check_size*(position.x()-aabb.min.x())))^int(floorf(check_size*(position.z()-aabb.min.z())))) &1) ? 0.8f : 0.2f;
            const Eigen::Vector3f floor_color = Eigen::Array3f{check*check*check, check*check, check};
            Eigen::Vector3f col = evaluate_shading(
                floor ? floor_color : scene.surface_color.array() * scene.surface_color.array(),
                light.ambient_color.array() * sky_color.array(),
                sun_color,
                floor ? 0.0f : brdf.metallic,
                floor ? 0.0f : brdf.subsurface,
                floor ? 1.0f : brdf.specular,
                floor ? 0.5f : brdf.roughness,
                0.0f,
                floor ? 0.0f : brdf.sheen,
                0.0f,
                floor ? 0.0f : brdf.clearcoat,
                brdf.clearcoat_gloss,
                light.parallel ? light.pos.normalized() : (light.pos - position).normalized(),
                -payload.dir.normalized(),
                normal);
            color = col.array();
        } break;
        case ERenderMode::NORMONLY: {
            Eigen::Vector3f n = normal;
            color = clamp((0.5 * n + Eigen::Vector3f::Constant(0.5)), 0.0f, 1.0f);
        } break;
        case ERenderMode::INSTANT_AO: {
            float col = powf(0.92f, payload.n_steps);
            color = Eigen::Vector3f::Constant(col);
        } break;
        case ERenderMode::POSITION: {
            color = clamp((position.normalized() * 0.5 + Eigen::Vector3f::Constant(0.5f)), 0.0f, 1.0f);
        } break;
        case ERenderMode::DEPTH: {
            float depth = camera_gaze.dot(position - camera_pos);
            color = Eigen::Vector3f::Constant(depth);
        }
        case ERenderMode::COST: {
            float col = (float)payload.n_steps / 30;
            color = Eigen::Array3f::Constant(col);
        }
        case ERenderMode::DISTANCE: {
            if(scene.slice_plane_z < 0){
                float distance = distances[i];
                if(distance > 1.0f){    // inside shows red
                    color = {1.0f, 0.0f, 0.0f};
                }else if(distance < -1.0f){ // outer shows blue
                    color = {0.0f, 0.0f, 1.0f};
                }else{  // near surface show white
                    color = Eigen::Vector3f::Constant(1.0f - std::abs(distance));
                }
            }else{
                float depth = camera_gaze.dot(position - camera_pos);
                color = Eigen::Vector3f::Constant(depth);
            }
        }
        case ERenderMode::SLICE: {
            if(scene.slice_plane_z < 0){
                float distance = distances[i];
                if(distance > 0.0f){    // outside shows faces
                    Eigen::Vector3f n = normal;
                    Eigen::Vector3f l = light.parallel ? light.pos.normalized() : (light.pos - position).normalized();
                    Eigen::Vector3f v = -payload.dir.normalized();
                    Eigen::Vector3f h = (v + l).normalized();
                
                    float spec = std::abs(h.dot(n));
                    spec = std::pow(spec, 16.0);

                    Eigen::Vector3f diffuse_color = light.light_color.array() * Eigen::Vector3f::Constant(light.kd * std::abs(l.dot(n))).array();
                    diffuse_color = clamp(light.ambient_color + diffuse_color, 0.0f, 1.0f);
                    Eigen::Vector3f specular_color = clamp((light.light_color * light.specular * spec), 0.0f, 1.0f);
                    
                    color = clamp(Eigen::Vector3f(scene.surface_color.array() * diffuse_color.array()) + specular_color, 0.0f, 1.0f);
                }else if(distance < 0.0f){ // inside show red
                    color = {1.0f, 0.0f, 0.0f};
                }
            }else{
                float depth = camera_gaze.dot(position - camera_pos);
                color = Eigen::Vector3f::Constant(depth);
            }
        }
    }
    renderbuffer[payload.idx] = {color.x(), color.y(), color.z(), 1.0f};
}

__global__ void limit_diff_pos(
    Eigen::Vector3f *origin,
    Eigen::Vector3f *f,
    Eigen::Vector3f *b,
    Eigen::Vector3f *l,
    Eigen::Vector3f *r,
    Eigen::Vector3f *u,
    Eigen::Vector3f *d,
    uint32_t num,
    float bias
){
    const uint32_t i = threadIdx.x + blockDim.x *  blockIdx.x;
    if(i >= num)return;
    f[i] = origin[i]; f[i].x() += bias;
    b[i] = origin[i]; b[i].x() -= bias;
    l[i] = origin[i]; l[i].y() -= bias;
    r[i] = origin[i]; r[i].y() += bias;
    u[i] = origin[i]; u[i].z() += bias;
    d[i] = origin[i]; d[i].z() -= bias;
}

__global__ void limit_diff_normal(
    Eigen::Vector3f *nor,
    float *fd,
    float *bd,
    float *ld,
    float *rd,
    float *ud,
    float *dd,
    uint32_t num
){
    const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= num)return;
    Eigen::Vector3f &normal = nor[i];
    normal.x() = (fd - bd);
    normal.y() = (rd - ld);
    normal.z() = (ud - dd);
    normal = normal.normalized();
}

__global__ void clear_buffer(Eigen::Array4f *buffer, uint32_t num, Eigen::Array4f background_color){
    const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= num)return;
    buffer[i] = {background_color.x(), background_color.y(), background_color.z(), background_color.w()};
}

__global__ void reverse_picture_verticle_kernel(Eigen::Array4f *buffer, Eigen::Vector2i shape){
    const uint32_t x = threadIdx.x;
    const uint32_t y = blockIdx.x;
    if(x >= shape.x() || y >= shape.y())return;
    const uint32_t idx = x + blockDim.x * y;
    uint32_t target_idx = shape.y() - y - 1;
    target_idx = target_idx * blockDim.x + x;
    Eigen::Array4f tmp = buffer[idx];
    buffer[idx] = buffer[target_idx];
    buffer[target_idx] = tmp;
}

void reverse_picture_verticle(GPUVector<Eigen::Array4f> &buffer, Eigen::Vector2i shape, cudaStream_t stream){
    reverse_picture_verticle_kernel<<<shape.y()/2, shape.x(), 0, stream>>>(buffer.ptr(), shape);
}

void clear_buffer(GPUVector<Eigen::Array4f> &buffer, Eigen::Array4f background_color, cudaStream_t stream){
    clear_buffer<<<div_round_up(buffer.size(), (size_t)128), 128, 0, stream>>>(
            buffer.ptr(), buffer.size(), background_color
        );
}

bool debug_project(const Eigen::Matrix<float, 4, 4>&proj, Eigen::Vector3f p, ImVec2& o) {
	Eigen::Vector4f ph; ph << p, 1.f;
	Eigen::Vector4f pa = proj * ph;
	if (pa.w() <= 0.f) return false;
	o.x = pa.x() / pa.w();
	o.y = pa.y() / pa.w();
	return true;
}

void add_debug_line(ImDrawList* list, const Eigen::Matrix<float, 4, 4>& proj, Eigen::Vector3f a, Eigen::Vector3f b, uint32_t col, float thickness = 1.0f) {
	ImVec2 aa, bb;
	if (debug_project(proj, a, aa) && debug_project(proj, b, bb)) {
		list->AddLine(aa, bb, col, thickness);
	}
}

void visualize_unit_cube(ImDrawList* list, const Eigen::Matrix<float, 4, 4>& world2proj, const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Matrix3f& render_aabb_to_local) {
	Eigen::Matrix3f m = render_aabb_to_local.transpose();
	add_debug_line(list, world2proj, m * Eigen::Vector3f{a.x(),a.y(),a.z()}, m * Eigen::Vector3f{a.x(),a.y(),b.z()}, 0xffff4040); // Z
	add_debug_line(list, world2proj, m * Eigen::Vector3f{b.x(),a.y(),a.z()}, m * Eigen::Vector3f{b.x(),a.y(),b.z()}, 0xffffffff);
	add_debug_line(list, world2proj, m * Eigen::Vector3f{a.x(),b.y(),a.z()}, m * Eigen::Vector3f{a.x(),b.y(),b.z()}, 0xffffffff);
	add_debug_line(list, world2proj, m * Eigen::Vector3f{b.x(),b.y(),a.z()}, m * Eigen::Vector3f{b.x(),b.y(),b.z()}, 0xffffffff);

	add_debug_line(list, world2proj, m * Eigen::Vector3f{a.x(),a.y(),a.z()}, m * Eigen::Vector3f{b.x(),a.y(),a.z()}, 0xff4040ff); // X
	add_debug_line(list, world2proj, m * Eigen::Vector3f{a.x(),b.y(),a.z()}, m * Eigen::Vector3f{b.x(),b.y(),a.z()}, 0xffffffff);
	add_debug_line(list, world2proj, m * Eigen::Vector3f{a.x(),a.y(),b.z()}, m * Eigen::Vector3f{b.x(),a.y(),b.z()}, 0xffffffff);
	add_debug_line(list, world2proj, m * Eigen::Vector3f{a.x(),b.y(),b.z()}, m * Eigen::Vector3f{b.x(),b.y(),b.z()}, 0xffffffff);

	add_debug_line(list, world2proj, m * Eigen::Vector3f{a.x(),a.y(),a.z()}, m * Eigen::Vector3f{a.x(),b.y(),a.z()}, 0xff40ff40); // Y
	add_debug_line(list, world2proj, m * Eigen::Vector3f{b.x(),a.y(),a.z()}, m * Eigen::Vector3f{b.x(),b.y(),a.z()}, 0xffffffff);
	add_debug_line(list, world2proj, m * Eigen::Vector3f{a.x(),a.y(),b.z()}, m * Eigen::Vector3f{a.x(),b.y(),b.z()}, 0xffffffff);
	add_debug_line(list, world2proj, m * Eigen::Vector3f{b.x(),a.y(),b.z()}, m * Eigen::Vector3f{b.x(),b.y(),b.z()}, 0xffffffff);
}

void SphereTracer::init_rays_from_camera(
            const Eigen::Vector2i &resolution,
            const Eigen::Vector2f &focal_length,
            const Eigen::Matrix<float, 3, 4> &camera_matrix,
            const Eigen::Vector2f &screen_center,
            const BoundingBox &aabb,
            float floor_y,
            float slice_plane_z,
            cudaStream_t stream){
    resize(resolution.x() * resolution.y());
    const dim3 threads = {16, 8, 1};
    const dim3 blocks = {div_round_up((uint32_t)resolution.x(), threads.x), div_round_up((uint32_t)resolution.y(), threads.y), 1};
    init_rays_from_camera_kernel<<<blocks, threads, 0, stream>>>(
        m_rays[0].pos.ptr(),
        m_rays[0].distance.ptr(),
        m_rays[0].payload.ptr(),
        resolution,
        focal_length,
        camera_matrix,
        screen_center,
        aabb,
        floor_y,
        slice_plane_z
    );
    m_n_rays_initialized = (uint32_t)resolution.x() * resolution.y();
}

void SphereTracer::init_rays_from_hit(
    RaysSDFSoa &rays_hit,
    uint32_t num,
    cudaStream_t stream
){
    resize(num);
    m_rays[0].memcpy_from(rays_hit, num, stream);
    m_n_rays_initialized = num;
}

void SphereTracer::init_rays_from_positions_and_directions(
    GPUVector<Eigen::Vector3f> &positions,
    GPUVector<Eigen::Vector3f> &directions,
    uint32_t num,
    BoundingBox aabb,
    cudaStream_t stream
){
    resize(num);
    m_rays[0].pos.memcpyfrom_device(positions.ptr(), num, stream);
    init_rays_from_positions_and_directions_kernel<<<div_round_up(num, 128u), 128, 0, stream>>>(
        directions.ptr(),
        m_rays[0].payload.ptr(),
        num
    );
    m_n_rays_initialized = num;
}

uint32_t SphereTracer::trace_bvh(TriangleBVH *bvh, Triangle *triangles, cudaStream_t stream){
    uint32_t n_alive = m_n_rays_initialized;
    m_n_rays_initialized = 0;
    if(!bvh)return 0;

    payload_direction_abuse_normal<<<div_round_up(n_alive, 128u), 128, 0, stream>>>(m_rays[0].nor.ptr(), m_rays[0].payload.ptr(), n_alive);
    cudaDeviceSynchronize();

    bvh->ray_trace_gpu(m_rays[0].pos.ptr(), m_rays[0].nor.ptr(), n_alive, triangles, stream);

    return n_alive;
}

uint32_t SphereTracer::trace_bvh_outer(TriangleBVH *bvh, Triangle *triangles, cudaStream_t stream){
    uint32_t n_alive = m_n_rays_initialized;
    m_n_rays_initialized = 0;
    if(!bvh)return 0;

    CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter.ptr(), 0, sizeof(uint32_t), stream));

    compact_outer_rays_kernel<<<div_round_up(n_alive, 128u), 128, 0, stream>>>(
        m_rays[0].pos.ptr(), m_rays[0].distance.ptr(), m_rays[0].payload.ptr(),
        m_rays_hit.pos.ptr(), m_rays_hit.distance.ptr(), m_rays_hit.payload.ptr(),
        n_alive, m_hit_counter.ptr()
    );

    CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_hit_counter.ptr(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    payload_direction_abuse_normal<<<div_round_up(n_alive, 128u), 128, 0, stream>>>(m_rays_hit.nor.ptr(), m_rays_hit.payload.ptr(), n_alive);
    cudaDeviceSynchronize();

    bvh->ray_trace_gpu(m_rays_hit.pos.ptr(), m_rays_hit.nor.ptr(), n_alive, triangles, stream);

    return n_alive;
}

uint32_t SphereTracer::trace(const distance_fun_t &distance_function, const BoundingBox &aabb, float distance_scale, float maximum_distance, cudaStream_t stream){
    if(m_n_rays_initialized == 0)return 0;
    CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter.ptr(), 0, sizeof(uint32_t), stream));
    const uint32_t STEPS_IN_BETWEEN_COMPACTION = 4;
    uint32_t n_alive = m_n_rays_initialized;
    m_n_rays_initialized = 0;
    uint32_t i = 1;
    uint32_t double_buffer_index = 0;
    while(i < 10000){
        uint32_t step_size = std::min(i, STEPS_IN_BETWEEN_COMPACTION);
        RaysSDFSoa &rays_current = m_rays[(double_buffer_index+1)%2];
        RaysSDFSoa &rays_tmp = m_rays[(double_buffer_index)%2];
        ++double_buffer_index;
        // Compact rays
        {
            CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter.ptr(), 0, sizeof(uint32_t), stream));
            
            compact_rays_kernel<<<div_round_up(n_alive, 128u), 128, 0, stream>>>(
                rays_tmp.pos.ptr(), rays_tmp.distance.ptr(), rays_tmp.payload.ptr(),
                rays_current.pos.ptr(), rays_current.distance.ptr(), rays_current.payload.ptr(),
                m_rays_hit.pos.ptr(), m_rays_hit.distance.ptr(), m_rays_hit.payload.ptr(),
                n_alive,
                aabb,
                m_alive_counter.ptr(), m_hit_counter.ptr()
            );
        }
        CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter.ptr(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        if(n_alive == 0)break;
        // sphere trace
        for(uint32_t j=0; j< step_size; ++j){
            distance_function(n_alive, rays_current.pos, rays_current.distance, stream);
            sdf_advance_position_kernel<<<div_round_up(n_alive, 128u), 128, 0, stream>>>(
                rays_current.pos.ptr(), rays_current.distance.ptr(), rays_current.payload.ptr(),
                nullptr, nullptr,
                n_alive,
                aabb,
                distance_scale,
                maximum_distance
            );
        }
        i+=step_size;
    }
    uint32_t n_hit;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter.ptr(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return n_hit;
}

bool SDFRenderer::build_bvh(std::string obj_path){
    Mesh::load_triangles(obj_path, m_triangles_cpu);

    m_bvh = TriangleBVH::create();
    clock_t start, end;
    start = clock();
    m_bvh->build(m_triangles_cpu, 8);
    end = clock();
    std::cout << "Building Triangle BVH Cost " << (float(end - start)/CLOCKS_PER_SEC) << " s." << std::endl;
    m_triangles_gpu.resize_and_memcpy_from_vector(m_triangles_cpu, render_stream);
    m_bvh->build_optix(m_triangles_gpu, render_stream);

    scene.aabb = BoundingBox(m_triangles_cpu.begin(), m_triangles_cpu.end());

    m_render_ground_truth = true;
    return true;
}

uint32_t SDFRenderer::trace_and_shade(Eigen::Vector2f focal_length){
    // define aabb
    BoundingBox aabb = scene.aabb;
    aabb.enlarge(scene.aabb_offset);

    // 1. clear buffer
    clear_buffer(m_frame_buffer.data_gpu, light.background_color, render_stream);

    // 2. init rays
    tracer.init_rays_from_camera(
        m_res, focal_length, camera.pose, {0.5, 0.5}, 
        aabb, 
        scene.floor_y, scene.slice_plane_z, 
        render_stream
    );
    
    // 3. trace
    m_n_hit = 0u;
    if(m_render_ground_truth && (m_render_mode != ERenderMode::SLICE)){
        m_n_hit = tracer.trace_bvh(m_bvh.get(), m_triangles_gpu.ptr(), render_stream);
    }

    if(m_render_mode == ERenderMode::SLICE && scene.slice_plane_z < 0){
        m_n_hit = tracer.n_rays_initialized();
        m_bvh->signed_distance_gpu(
            tracer.rays_init().pos.ptr(), 
            tracer.rays_init().distance.ptr(), 
            m_n_hit, 
            m_triangles_gpu.ptr(),
            m_sdf_calculate_mode,
            false,
            render_stream
        );

        m_n_hit = tracer.trace_bvh_outer(m_bvh.get(), m_triangles_gpu.ptr(), render_stream);
    }
    
    RaysSDFSoa &rays = m_render_ground_truth && m_render_mode != ERenderMode::SLICE ? tracer.rays_init() : tracer.rays_hit();

    if(m_render_mode == ERenderMode::Shade){    // prepare shadow rays
        shadow_tracer.init_rays_from_hit(rays, m_n_hit, render_stream); // shadow rays start from hit point
        shadow_tracer.set_trace_shadow_rays(true);
        shadow_tracer.set_shadow_sharpness(m_shadow_sharpness);
        RaysSDFSoa &shadow_rays_init = shadow_tracer.rays_init();
        prepare_shadow_rays<<<div_round_up(m_n_hit, 128u), 128, 0, render_stream>>>(
            shadow_rays_init.pos.ptr(),
            shadow_rays_init.nor.ptr(),
            shadow_rays_init.distance.ptr(),
            shadow_rays_init.prev_distance.ptr(),
            shadow_rays_init.total_distance.ptr(),
            shadow_rays_init.min_visibility.ptr(),
            shadow_rays_init.payload.ptr(),
            m_n_hit,
            aabb,
            light.pos,
            light.parallel
        );
        uint32_t n_hit_shadow = 0;
        if(m_render_ground_truth)n_hit_shadow = shadow_tracer.trace_bvh(m_bvh.get(), m_triangles_gpu.ptr(), render_stream);

        auto &shadow_rays_hit = m_render_ground_truth ? shadow_tracer.rays_init() : shadow_tracer.rays_hit();

        write_shadow_ray_result<<<div_round_up(n_hit_shadow, 128u), 128, 0, render_stream>>>(
            shadow_rays_hit.pos.ptr(),
            shadow_rays_hit.payload.ptr(),
            shadow_rays_hit.min_visibility.ptr(),
            rays.distance.ptr(),
            n_hit_shadow,
            aabb
        );
    }

    // 4. shade
    shade_sdf_kernel<<<div_round_up(m_n_hit, 128u), 128, 0, render_stream>>>(
        rays.pos.ptr(),
        rays.nor.ptr(),
        rays.distance.ptr(),
        rays.payload.ptr(),
        m_frame_buffer.gpu(),
        m_n_hit,
        scene.aabb,
        m_render_mode,
        camera.pose,
        light,
        brdf,
        scene
    );
    return m_n_hit;
}

bool SDFRenderer::frame(){
    clock_t start, end;
    start = clock();
    Eigen::Vector2f focal_length = camera.calc_focal_length(m_res);
    trace_and_shade(focal_length);

    if(m_gui){
        draw_gui();
    }
    end = clock();
    elapse = (float(end - start)/CLOCKS_PER_SEC)*1000.0f;
    fps = 1000.0f / elapse;

    return !glfwWindowShouldClose(m_window);
}

std::vector<Eigen::Array4f> SDFRenderer::render_ray_trace(
    const std::vector<Eigen::Vector3f> &cam_pos = std::vector<Eigen::Vector3f>(), 
    Eigen::Vector3f cam_focus = Eigen::Vector3f::Zero(), 
    ERenderMode render_mode = ERenderMode::LATTICE
    ){
    if(m_gui){
        init_window(m_res);

        coord = std::unique_ptr<Coordinate>(new Coordinate());
        light_sphere = std::unique_ptr<LightSphere>(new LightSphere());
        if(cam_pos.empty()){
            while(frame());
        }
        else if(cam_pos.size() == 1){
            camera.pose.col(3) == cam_pos[0];
            camera.focus(cam_focus);
            while(frame());
        }else{
            camera.pose.col(3) == cam_pos[0];
            camera.focus(cam_focus);
            float t = 1e-3f;
            while(frame()){
                int i = (int)std::floor(t) % cam_pos.size();
                int n = (i == cam_pos.size() ? 0 : i+1);
                float alpha = fractf(t);
                camera.pose.col(3) = (1 - alpha) * cam_pos[i] + alpha * cam_pos[n];
                camera.focus(cam_focus);
                t += amplitude * 1e-2f;
            }
        }
        return std::vector<Eigen::Array4f>();
    }else{
        if(cam_pos.empty()){
            throw std::runtime_error("Camera positions can not be empty");
        }else{
            if(cam_pos.size() == 1){
                // 1. set camera
                camera.pose.col(3) = cam_pos[0];
                camera.focus(cam_focus);
                // 2. render
                frame();
                // 3. return picture
                cudaDeviceSynchronize();
                m_frame_buffer.gpu2cpu();
                return m_frame_buffer.data_cpu;
            }else{
                // allocate memory
                std::vector<Eigen::Array4f> pics(m_frame_buffer.data_cpu.size() * cam_pos.size());

                for(int i=0; i<cam_pos.size(); i++){
                    // 1. set camera
                    camera.pose.col(3) = cam_pos[i];
                    camera.focus(cam_focus);
                    // 2. render
                    frame();
                    // 3. return picture
                    cudaDeviceSynchronize();

                    m_frame_buffer.data_gpu.memcpyto(&pics[0], m_frame_buffer.data_gpu.size(), render_stream);
                }
                return pics;
            }
        }
    }
}

std::vector<Eigen::Array4f> SDFRenderer::read_and_render_frame(
    std::vector<Eigen::Vector3f> &points,
    std::vector<Eigen::Vector3f> &normals,
    std::vector<Eigen::Vector<int, 1>> &hit,
    std::vector<Eigen::Vector<int, 1>> &n_steps,
    std::vector<Eigen::Vector<float, 1>> &distances,
    Eigen::Vector3f pos,
    Eigen::Vector3f to){
    
    // 1. read payloads
    size_t n_pixels = m_res.x() * m_res.y();
    tracer.resize(n_pixels);
    std::vector<Eigen::Vector3f> pos_hit(n_pixels);
    std::vector<Eigen::Vector3f> nor_hit(n_pixels);
    std::vector<float> distance_hit(n_pixels);
    std::vector<SDFPayload> payloads_hit(n_pixels);
    uint32_t n_hit = 0;

    camera.pose.col(3) = pos;
    camera.focus(to);
    Eigen::Vector2f focal_length = camera.calc_focal_length(m_res);

    for(int i=0; i<m_res.x(); i++){
        for(int j=0; j<m_res.y(); j++){
            uint32_t idx = i * m_res.y() + j;
            if(hit[idx].x()){
                pos_hit[n_hit] = points[idx];
                nor_hit[n_hit] = normals[idx];
                distance_hit[n_hit] = distances[idx].x();
                Eigen::Vector2i pixel = {i, j};
                Eigen::Vector3f dir = init_rays_direction(pixel, m_res, focal_length, camera.pose, {0.5, 0.5});
                payloads_hit[n_hit].dir = dir;
                payloads_hit[n_hit].idx = idx;
                payloads_hit[n_hit].dir = dir;
                n_hit++;
            }
        }
    }

    tracer.rays_hit().pos.memcpyfrom(&pos_hit[0], n_hit, render_stream);
    tracer.rays_hit().nor.memcpyfrom(&nor_hit[0], n_hit, render_stream);
    tracer.rays_hit().distance.memcpyfrom(&distance_hit[0], n_hit, render_stream);
    tracer.rays_hit().payload.memcpyfrom(&payloads_hit[0], n_hit, render_stream);
    
    scene.aabb = BoundingBox(Eigen::Vector3f::Constant(-1.0f), Eigen::Vector3f::Constant(1.0f));
    m_read_frame = true;

    if(m_gui){
        init_window(m_res);
        while(!glfwWindowShouldClose(m_window)){
            clock_t start, end;
            start = clock();

            clear_buffer(m_frame_buffer.data_gpu, light.background_color, render_stream);

            shade_sdf_kernel<<<div_round_up(n_hit, 128u), 128, 0, render_stream>>>(
                tracer.rays_hit().pos.ptr(), tracer.rays_hit().nor.ptr(), tracer.rays_hit().distance.ptr(), tracer.rays_hit().payload.ptr(),
                m_frame_buffer.gpu(), 
                n_hit, 
                scene.aabb, 
                m_render_mode,
                camera.pose,
                light,
                brdf,
                scene
            );
            cudaDeviceSynchronize();
            m_frame_buffer.gpu2cpu();

            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            GLClearError();
            glDrawPixels((uint32_t)m_res.x(), (uint32_t)m_res.y(), GL_RGBA, GL_FLOAT, (void*)m_frame_buffer.cpu());
            GLCheckError();

            imgui_draw();

            glfwSwapBuffers(m_window);
            glfwPollEvents();

            keyboard_event_handler();
            cursor_event_handler();
            end = clock();
            elapse = (float(end - start)/CLOCKS_PER_SEC)*1000.0f;
            fps = 1000.0f / elapse;
        }
        return std::vector<Eigen::Array4f>();
    }else{
        clear_buffer(m_frame_buffer.data_gpu, light.background_color, render_stream);

        shade_sdf_kernel<<<div_round_up(n_hit, 128u), 128, 0, render_stream>>>(
            tracer.rays_hit().pos.ptr(), tracer.rays_hit().nor.ptr(), tracer.rays_hit().distance.ptr(), tracer.rays_hit().payload.ptr(),
            m_frame_buffer.gpu(), 
            n_hit, 
            scene.aabb, 
            m_render_mode,
            camera.pose,
            light,
            brdf,
            scene
        );
        cudaDeviceSynchronize();
        m_frame_buffer.gpu2cpu();
        return m_frame_buffer.data_cpu;
    }
}

bool SDFRenderer::init_window(Eigen::Vector2i win_res){
    // initialize window
    if(!glfwInit()){
        std::cout << "Error: Can not initialize glfw! " << std::endl;
        return 0;
    }
    m_window = glfwCreateWindow(win_res.x(), win_res.y(), "geolab", NULL, NULL);
    if(!m_window){
        glfwTerminate();
        std::cout << "Error: Can not create window! " << std::endl;
        return 0;
    }
  
    glfwMakeContextCurrent(m_window);
  
    //should be done after glfw initialized
    if(glewInit() != GLEW_OK){
        std::cout << "Error: Can not initialize glew" << std::endl;
        return 0;
    }
    std::cout << glGetString(GL_VERSION) << std::endl;

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigInputTrickleEventQueue = false;
    // Setup Platform/Renderer bindings
    if(!ImGui_ImplGlfw_InitForOpenGL(m_window, true))return 0;
    if(!ImGui_ImplOpenGL3_Init("#version 330 core"))return 0;
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    return true;
}

void SDFRenderer::draw_gui(){
    cudaDeviceSynchronize();
    m_frame_buffer.gpu2cpu();

    auto transform = mvp(camera.pose, 0.1f, 50.0f, camera.fov.x(), 1.0f);

    GL_CHECK(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    GL_CHECK(glDrawPixels((uint32_t)m_res.x(), (uint32_t)m_res.y(), GL_RGBA, GL_FLOAT, (void*)m_frame_buffer.cpu()));

    if(draw_coordinate_axis){
        coord->update_uniform(transform);
        coord->draw();
    }
    if(draw_light_sphere){
        float radius = 1e-1f;
        light_sphere->update_uniform(transform, light.light_color, light.pos, radius);
        light_sphere->draw();
    }

    imgui_draw();

    GL_CHECK(glfwSwapBuffers(m_window));
    GL_CHECK(glfwPollEvents());

    keyboard_event_handler();
    cursor_event_handler();
}

// IMGUI Drawer
void SDFRenderer::imgui_draw(){
    ImGui::SetNextWindowSize(ImVec2(640, 480));

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Setting");

    imgui_general_draw();

    if(ImGui::CollapsingHeader("Camera Setting")){
        imgui_camera_draw();
    }

    if(ImGui::CollapsingHeader("Light Setting")){
        imgui_light_draw();
    }

    if(ImGui::CollapsingHeader("Scene Setting")){
        imgui_scene_draw();
    }

    if(ImGui::CollapsingHeader("BRDF Setting")){
        imgui_brdf_draw();
    }

    if(ImGui::CollapsingHeader("SDF Rendering Setting")){
        imgui_sdf_render_draw();
    }

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void SDFRenderer::imgui_general_draw(){
    ImGui::Text("elapse: %.3f ms fps: %.3f", elapse, fps);
    ImGui::Text("hit rays: %d", m_n_hit);
    ImGui::SliderFloat("amplitude", &amplitude, 0.0f, 1.0f);
    ImGui::Combo("Render Mode", (int*)&m_render_mode, RenderModeStr);
    ImGui::Checkbox("draw_coordinate_axis", &draw_coordinate_axis);
}

void SDFRenderer::imgui_camera_draw(){
    if(ImGui::Button("Focus object")){
        camera.focus();
    }
    ImGui::SameLine();
    if(ImGui::Button("Reset Camera")){
        camera.pose.col(3) = Eigen::Vector3f(0.0f, 0.0f, 4.0f);
        camera.focus();
    }
    ImGui::BeginTable("Camera Pose", 4);
    ImGui::TableNextRow();  // side up gaze pos
    ImGui::TableNextColumn(); ImGui::Text("side");
    ImGui::TableNextColumn(); ImGui::Text("up"); 
    ImGui::TableNextColumn(); ImGui::Text("gaze");
    ImGui::TableNextColumn(); ImGui::Text("pos");
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(0,0));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(0,1));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(0,2));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(0,3));
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(1,0));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(1,1));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(1,2));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(1,3));
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(2,0));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(2,1));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(2,2));
    ImGui::TableNextColumn(); ImGui::Text("%.3f",camera.pose(2,3));
    ImGui::EndTable();
}

void SDFRenderer::imgui_light_draw(){
    ImGui::Checkbox("draw_light_sphere", &draw_light_sphere);
    ImGui::BeginTable("Light Pos", 3);
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::InputFloat("x", &light.pos.x());
    ImGui::TableNextColumn(); ImGui::InputFloat("y", &light.pos.y());
    ImGui::TableNextColumn(); ImGui::InputFloat("z", &light.pos.z());
    ImGui::EndTable();
    ImGui::ColorEdit3("Ambient Color", &light.ambient_color.x());
    ImGui::ColorEdit4("Background Color", &light.background_color.x());
    ImGui::ColorEdit3("Light Color", &light.light_color.x());
    ImGui::SliderFloat("kd", &light.kd, 0.0f, 2.0f);
    ImGui::SliderFloat("specular", &light.specular, 0.0f, 10.0f);
    ImGui::Checkbox("Parallel Light", &light.parallel);
}

void SDFRenderer::imgui_brdf_draw(){
    ImGui::SliderFloat("metallic", &brdf.metallic, 0.0f, 2.0f);
    ImGui::SliderFloat("subsurface", &brdf.subsurface, 0.0f, 2.0f);
    ImGui::SliderFloat("specular", &brdf.specular, 0.0f, 2.0f);
    ImGui::SliderFloat("roughness", &brdf.roughness, 0.0f, 2.0f);
    ImGui::SliderFloat("sheen", &brdf.sheen, 0.0f, 2.0f);
    ImGui::SliderFloat("clearcoat", &brdf.clearcoat, 0.0f, 2.0f);
    ImGui::SliderFloat("clearcoat_gloss", &brdf.clearcoat_gloss, 0.0f, 2.0f);
}

void SDFRenderer::imgui_scene_draw(){
    ImGui::ColorEdit3("Surface Color", &scene.surface_color.x());
    ImGui::ColorEdit3("Sky Color", &scene.sky_color.x());
    ImGui::SliderFloat("slice_plane_z", &scene.slice_plane_z, -10.0f, 2.0f);
    ImGui::SliderFloat("floor_y", &scene.floor_y, -2.0f, 2.0f);
    ImGui::SliderFloat("aabb offset", &scene.aabb_offset, -1.0f, 1.0f);
    ImGui::BeginTable("AABB", 4);
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::Text("x");
    ImGui::TableNextColumn(); ImGui::Text("y");
    ImGui::TableNextColumn(); ImGui::Text("z");
    ImGui::TableNextColumn(); ImGui::Text(" ");
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::InputFloat("x", &scene.aabb.min.x());
    ImGui::TableNextColumn(); ImGui::InputFloat("y", &scene.aabb.min.y());
    ImGui::TableNextColumn(); ImGui::InputFloat("z", &scene.aabb.min.z());
    ImGui::TableNextColumn(); ImGui::Text("min");
    ImGui::TableNextRow();
    ImGui::TableNextColumn(); ImGui::InputFloat("x", &scene.aabb.max.x());
    ImGui::TableNextColumn(); ImGui::InputFloat("y", &scene.aabb.max.y());
    ImGui::TableNextColumn(); ImGui::InputFloat("z", &scene.aabb.max.z());
    ImGui::TableNextColumn(); ImGui::Text("max");
    ImGui::EndTable();
}

void SDFRenderer::imgui_sdf_render_draw(){
    ImGui::Combo("SDF Calculate Mode", (int*)&m_sdf_calculate_mode, SDFCalcModeStr);
}

// Window Message Controller

void SDFRenderer::mouse_drag(ImVec2 rel, int button){
    bool is_left_held = (button & 1) != 0;
    Eigen::Vector3f up = camera.view_up();
    Eigen::Vector3f side = camera.view_side();

	if (is_left_held) {
        Eigen::Vector3f pos = camera.view_pos();
		float alpha = amplitude * rel.x / PI * 180;
        float beta = amplitude * rel.y / PI * 180;
        Eigen::Matrix3f translate =
		(Eigen::AngleAxisf(static_cast<float>(-amplitude * 0.0001 * rel.x / PI * 180), up) * // Scroll sideways around up vector
		Eigen::AngleAxisf(static_cast<float>(-amplitude * 0.0001 * rel.y / PI * 180), side)).matrix(); // Scroll around side vector
        camera.pose = translate * camera.pose;
	}
}

void SDFRenderer::mouse_scroll(float delta){
    if(delta == 0.0f)return;
    
    if(!ImGui::GetIO().WantCaptureMouse){
        camera.pose.col(3) += delta*amplitude * camera.pose.col(2);
    }
}

bool SDFRenderer::cursor_event_handler(){
    ImVec2 m = ImGui::GetMousePos();
	int mb = 0;
	float mw = 0.f;
	ImVec2 relm = {};
    if (!ImGui::IsAnyItemActive() && !ImGuizmo::IsUsing() && !ImGuizmo::IsOver()) {
        mw = ImGui::GetIO().MouseWheel;
		relm = ImGui::GetIO().MouseDelta;
		if (ImGui::GetIO().MouseDown[0]) mb |= 1;
		if (ImGui::GetIO().MouseDown[1]) mb |= 2;
		if (ImGui::GetIO().MouseDown[2]) mb |= 4;
		
	}
	mouse_scroll(mw);
	mouse_drag(relm, mb);
    return 1;
}

bool SDFRenderer::keyboard_event_handler(){
    if (ImGui::GetIO().WantCaptureKeyboard) {
		return false;
	}
    if (ImGui::IsKeyPressed('W')) {
        camera.pose.col(3) += amplitude * camera.pose.col(2);
	}
    if (ImGui::IsKeyPressed('S')) {
        camera.pose.col(3) -= amplitude * camera.pose.col(2);
	}
    if (ImGui::IsKeyPressed('A')) {
        camera.pose.col(3) -= amplitude * camera.pose.col(0);
	}
    if (ImGui::IsKeyPressed('D')) {
        camera.pose.col(3) += amplitude * camera.pose.col(0);
	}
    if (ImGui::IsKeyPressed(ImGuiKey_::ImGuiKey_UpArrow)){
        camera.pose.col(3) -= amplitude * camera.pose.col(1);
    }
    if (ImGui::IsKeyPressed(ImGuiKey_::ImGuiKey_DownArrow)){
        camera.pose.col(3) -= amplitude * camera.pose.col(1);
    }
    return true;
}

