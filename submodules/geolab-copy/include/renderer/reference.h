#pragma once
#ifndef REFERENCE_H
#define REFERENCE_H

#include<renderer/m_glfw_imgui_utils.cuh>
#include<geometry/polygon/icosphere.h>

class Coordinate : public Drawable{
    private:
        // uniforms
        uint32_t uniform_world;
        Eigen::Matrix4f m_world;

    public:
        Coordinate(): Drawable(BASIC_SHADER){
            // vertex 12 lines * 2 vertices * 6 attribute = 144 numbers
            const float axis[144] = {
                // 1
                -1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f,
                1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f,
                // 2
                -1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
                -1.0f,  1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
                // 3
                -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
                -1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
                // 4
                1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                -1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                // 5
                1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, -1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                // 6
                1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                1.0f,  1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
                // 7
                1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
                1.0f,  1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
                // 8
                1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, -1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                // 9
                -1.0f,  1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
                -1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                // 10
                -1.0f,  1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
                1.0f,  1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
                // 11
                -1.0f, -1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, -1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                // 12
                -1.0f, -1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                -1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f,
            };

            GL_CHECK(glGenBuffers(1, &VB));
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, VB));
            GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(axis), axis, GL_STATIC_DRAW));
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
            // bind uniforms
            uniform_world = shader.bind_uniform("world");
        }
        ~Coordinate(){ if(VB)GL_CHECK(glDeleteBuffers(1, &VB)); }
        void update_uniform(Eigen::Matrix4f &matrix){
            m_world = matrix;
        }
        void draw() override{
            shader.bind();
            // update uniform variable
            GL_CHECK(glUniformMatrix4fv(uniform_world, 1, GL_FALSE, m_world.data()));
            // select buffer
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, VB));

            GL_CHECK(glEnableVertexAttribArray(0));
            GL_CHECK(glEnableVertexAttribArray(1));
            GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0));
            GL_CHECK(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3*sizeof(float))));

            GL_CHECK(glDrawArrays(GL_LINES, 0, 24));
            // unbind buffer
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
            shader.unbind();
        }
};

// use level 3 icosphere
class LightSphere: public Drawable{
    private:
        // uniforms 
        uint32_t uniform_world, uniform_color;
        Eigen::Matrix4f m_world = Eigen::Matrix4f::Identity();
        Eigen::Vector4f m_color = Eigen::Vector4f::Ones();
        Eigen::Vector3f m_center = Eigen::Vector3f::Zero();
        float m_radius = 1.0f;

    
    public:
        LightSphere(): Drawable(LIGHT_SPHERE_SHADER){
            auto sphere_mesh = IcoSphere().triangulate();
            GL_CHECK(glGenBuffers(1, &VB));
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, VB));
            GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sphere_mesh->bytes_v(), (void*)sphere_mesh->data_v(), GL_STATIC_DRAW));
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

            GL_CHECK(glGenBuffers(1, &EB));
            GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EB));
            GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_mesh->bytes_f(), (void*)sphere_mesh->data_f(), GL_STATIC_DRAW));
            GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
            // bind uniforms
            uniform_world = shader.bind_uniform("world");
            uniform_color = shader.bind_uniform("in_color");
        }
        ~LightSphere(){
            if(VB)GL_CHECK(glDeleteBuffers(1, &VB));
            if(EB)GL_CHECK(glDeleteBuffers(1, &EB));
        }
        void update_uniform(const Eigen::Matrix4f &matrix, const Eigen::Vector3f &color, const Eigen::Vector3f &center, const float &radius){
            m_world = matrix;
            m_color = Eigen::Vector4f(color.x(), color.y(), color.z(), 1.0f);
            m_center = center;
            m_radius = radius;
        }
        void draw() override{
            Eigen::Matrix4f t = m_world * translate(m_center);
            t = t * scale(m_radius);

            shader.bind();
            GL_CHECK(glUniformMatrix4fv(uniform_world, 1, GL_FALSE, t.data()));
            GL_CHECK(glUniform4f(uniform_color, m_color.x(), m_color.y(), m_color.z(), m_color.w()));

            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, VB));
            GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EB));

            GL_CHECK(glEnableVertexAttribArray(0)); 
            GL_CHECK(glEnableVertexAttribArray(1));
            GL_CHECK(glEnableVertexAttribArray(2));
            GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0));
            GL_CHECK(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3*sizeof(float))));
            GL_CHECK(glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6*sizeof(float))));

            GL_CHECK(glDrawElements(GL_TRIANGLES, 20 * 3 * 4 * 4 * 4, GL_UNSIGNED_INT, nullptr));

            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
            GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
            shader.unbind();
        }

    private:
        Eigen::Matrix4f translate(Eigen::Vector3f t){
            Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
            translation.col(3) = Eigen::Vector4f(t.x(), t.y(), t.z(), 1.0f);
            return translation;
        }
        Eigen::Matrix4f scale(float r){
            Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
            translation(0,0) = r;
            translation(1,1) = r;
            translation(2,2) = r;
            translation(3,3) = 1.0f;
            return translation;
        }
};

#endif