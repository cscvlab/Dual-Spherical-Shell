#pragma once
#ifndef GLRENDERER
#define GLRENDERER

#include<iostream>
#include<vector>
#include<Eigen/Eigen>
#include<array>

#include<renderer/camera.cuh>
#include<renderer/scene.cuh>
#include<renderer/m_glfw_imgui_utils.cuh>
#include<renderer/reference.h>

#define EDGE_SHADER_PATH "./data/shaders/polygon_edge.shader"
#define POLYGON_SHADER_PATH "./data/shaders/polygon.shader"

template<uint32_t STRIDE>
class PolygonFaceDrawable : public Drawable{
    private:
        uint32_t m_size_v;
        uint32_t m_size_f;
        uint32_t uniform_world;
        Eigen::Matrix4f m_world;

    public:
        PolygonFaceDrawable(Polygon<STRIDE> &p): Drawable(POLYGON_SHADER_PATH){
            auto mesh = p.triangulate();
            m_size_v = mesh->size_v();
            m_size_f = mesh->size_f();
            
            GL_CHECK(glGenBuffers(1, &VB));
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, VB));
            GL_CHECK(glBufferData(GL_ARRAY_BUFFER, mesh->bytes_v(), (void*)mesh->data_v(), GL_STATIC_DRAW));
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

            GL_CHECK(glGenBuffers(1, &EB));
            GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EB));
            GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->bytes_f(), (void*)mesh->data_f(), GL_STATIC_DRAW));
            GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));

            uniform_world = shader.bind_uniform("world");
        }    
        ~PolygonFaceDrawable(){
            if(VB)glDeleteBuffers(1, &VB);
            if(EB)glDeleteBuffers(1, &EB);
        }

        void update_uniform(Eigen::Matrix4f &transform){ m_world = transform; }

        void draw() override{
            shader.bind();
            GL_CHECK(glUniformMatrix4fv(uniform.world, 1, GL_FALSE, m_world.data()));
            
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, VB));
            GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EB));

            GL_CHECK(glEnableVertexAttribArray(0));
            GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0));
            GL_CHECK(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3*sizeof(float))));
            GL_CHECK(glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6*sizeof(float))));

            GL_CHECK(glDrawElements(GL_TRIANGLES, m_size_f, GL_UNSIGNED_INT, nullptr));

            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
            GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
            shader.unbind();
        }
  
};

class PolygonRenderer{
    public:
        Camera camera;
        Light light;
        BRDFParams brdf;
        Scene scene;
        // bool m_gui = false;
        bool draw_coordinate_axis = false;
        bool draw_light_sphere = false;

    private:
        Eigen::Vector2i m_res = {1920, 1080};

        // resource
        std::unique_ptr<Texture> m_texture;
        GLShader m_edge_shader;
        GLShader m_polygon_shader;
        std::vector<uint32_t> buffers;
        GLFWwindow *m_window;
        float fps = 0.0f;
        float elapse = 0.0f;
        float amplitude = 0.5f;

        // auxiliary object
        std::unique_ptr<Coordinate> coord;
        std::unique_ptr<LightSphere> light_sphere;

    private:
        bool init_window(Eigen::Vector2i win_res);
        void mouse_drag(ImVec2 rel, int button);
        void mouse_scroll(float delta);
        bool cursor_event_handler();
        bool keyboard_event_handler();
        // imgui
        void imgui_draw();
        void imgui_general_draw();
        void imgui_camera_draw();
        void imgui_scene_draw();
        // loop function
        void draw_gui(Drawable &drawable);
        bool frame(Drawable &drawable);

    public:
        PolygonRenderer(Eigen::Vector2i win_res = {1920, 1080}): m_res(win_res), m_edge_shader(EDGE_SHADER_PATH), m_polygon_shader(POLYGON_SHADER_PATH){}
        ~PolygonRenderer(){}
        void render(Drawable &polygon);
};

#endif