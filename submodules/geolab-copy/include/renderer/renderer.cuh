#pragma once
#ifndef RENDERER
#define RENDERER


#include<iostream>

#include<utils/gpubuffer.cuh>
#include<geometry/trianglebvh.cuh>
#include<geometry/sdf.cuh>
#include<geometry/mesh.h>
#include<renderer/camera.cuh>
#include<renderer/scene.cuh>

#include<Eigen/Eigen>

#include<renderer/m_glfw_imgui_utils.cuh>
#include<renderer/reference.h>
    
enum class ERenderMode {
    Shade,
    INSTANT_AO,
    NORMONLY,
    LATTICE,
    DIFFUSE,
    POSITION,
    DEPTH,
    COST,
    DISTANCE,
    SLICE
};

static constexpr const char* RenderModeStr = "Shade\0INSTANT_AO\0NORMONLY\0LATTICE\0DIFFUSE\0POSITION\0DEPTH\0COST\0DISTANCE\0SLICE\0\0";
static constexpr const char* SDFCalcModeStr = "WATERTIGHT\0RAYSTAB\0PATHESCAPE\0\0";

using distance_fun_t = std::function<void(uint32_t, GPUVector<Eigen::Vector3f>&, GPUVector<float>&, cudaStream_t)>;
using normal_fun_t = std::function<void(uint32_t, GPUVector<Eigen::Vector3f>&, GPUVector<Eigen::Vector3f>&, cudaStream_t)>;

class SphereTracer{
    public:
        SphereTracer(): m_hit_counter(1), m_alive_counter(1){}

        void init_rays_from_camera(
            const Eigen::Vector2i &resolution,
            const Eigen::Vector2f &focal_length,
            const Eigen::Matrix<float, 3, 4> &camera_matrix,
            const Eigen::Vector2f &screen_center,
            const BoundingBox &aabb,
            float floor_y,
            float slice_plane_z,
            cudaStream_t stream
        );
        // This method used when trace shadow rays
        void init_rays_from_hit(
            RaysSDFSoa &rays_hit,
            uint32_t num,
            cudaStream_t stream
        );
        void init_rays_from_positions_and_directions(
            GPUVector<Eigen::Vector3f> &positions,
            GPUVector<Eigen::Vector3f> &directions,
            const uint32_t num,
            const BoundingBox aabb,
            cudaStream_t stream
        );

        // raytrace
        uint32_t trace_bvh(TriangleBVH *bvh, Triangle *triangles, cudaStream_t stream);
        // available while ERenderMode == slice
        uint32_t trace_bvh_outer(TriangleBVH *bvh, Triangle *triangles, cudaStream_t stream);
        // sphere trace
        uint32_t trace(const distance_fun_t &distance_function, const BoundingBox &aabb, float distance_scale, float maximum_distance, cudaStream_t stream);

        void resize(size_t num){
            m_rays[0].resize(num);
            m_rays[1].resize(num);
            m_rays_hit.resize(num);
        }
        RaysSDFSoa& rays_hit(){return m_rays_hit;}
        RaysSDFSoa& rays_init(){return m_rays[0];}
        uint32_t n_rays_initialized() const {return m_n_rays_initialized;}
        void set_trace_shadow_rays(bool val) { m_trace_shadow_rays = val; }
        void set_shadow_sharpness(float val) { m_shadow_sharpness = val;}

    private:
        RaysSDFSoa m_rays[2];
        RaysSDFSoa m_rays_hit;
        GPUVector<uint32_t> m_hit_counter;
        GPUVector<uint32_t> m_alive_counter;
        uint32_t m_n_rays_initialized = 0;
        bool m_trace_shadow_rays = false;
        bool m_shadow_sharpness = 2048.f;
};

class SDFRenderer{
    public:
        Camera camera;
        Light light;
        BRDFParams brdf;
        Scene scene;
    private:
        Eigen::Vector2i m_res = {1920, 1080};
        GPUBuffer<Eigen::Array4f, 2> m_frame_buffer;
        Texture m_texture;
        SphereTracer tracer;
        SphereTracer shadow_tracer;
        cudaStream_t render_stream;

        GLFWwindow *m_window;
        uint32_t m_n_hit = 0u;
        float fps = 0.0f;
        float elapse = 0.0f;
        float amplitude = 0.5f;
        float m_shadow_sharpness = 2048.0f;
        std::unique_ptr<Coordinate> coord;
        std::unique_ptr<LightSphere> light_sphere;
        
        
    private:
        std::shared_ptr<TriangleBVH> m_bvh;   // available when trace gt
        std::vector<Triangle> m_triangles_cpu;
        GPUVector<Triangle> m_triangles_gpu;

    public:
        bool m_gui = false;
        bool m_read_frame = false;
        SDFCalcMode m_sdf_calculate_mode = SDFCalcMode::RAYSTAB;
        ERenderMode m_render_mode = ERenderMode::LATTICE;
        bool m_render_ground_truth = false;
        bool draw_coordinate_axis = false;  
        bool draw_light_sphere = false;

    private:
        bool init_window(Eigen::Vector2i win_res);
        void mouse_drag(ImVec2 rel, int button);
        void mouse_scroll(float delta);
        bool cursor_event_handler();
        bool keyboard_event_handler();

        void draw_gui();

        void imgui_draw();
        void imgui_general_draw();
        void imgui_camera_draw();
        void imgui_brdf_draw();
        void imgui_scene_draw();
        void imgui_light_draw();
        void imgui_sdf_render_draw();


    public:
        SDFRenderer(Eigen::Vector2i win_res = {1920, 1080}, bool gui = false): m_res(win_res), m_frame_buffer(win_res), m_gui(gui){
            CUDA_CHECK_THROW(cudaStreamCreate(&render_stream));
        }

        bool build_bvh(std::string obj_path);

        /** m_gui = true , show animation 
        * m_gui = false , output images
        * cam_pos.size() == 0 user control
        * cam_pos.size() == 1 init camera pos
        * cam_pos.size() > 1 auto animation
        */
        std::vector<Eigen::Array4f> render_ray_trace(
            const std::vector<Eigen::Vector3f> &cam_pos, 
            Eigen::Vector3f cam_focus, 
            ERenderMode render_mode 
        ); 

        std::vector<Eigen::Array4f> read_and_render_frame(
            std::vector<Eigen::Vector3f> &points,
            std::vector<Eigen::Vector3f> &normals,
            std::vector<Eigen::Vector<int, 1>> &hit,
            std::vector<Eigen::Vector<int, 1>> &n_steps,
            std::vector<Eigen::Vector<float, 1>> &distances,
            Eigen::Vector3f pos,
            Eigen::Vector3f to
        );


    private:
        // loop function
        bool frame();   // draw and message
        uint32_t trace_and_shade(Eigen::Vector2f focal_length);   // trace and shade
        void draw_ray_trace(); 
        void draw_sphere_trace();

};

#endif