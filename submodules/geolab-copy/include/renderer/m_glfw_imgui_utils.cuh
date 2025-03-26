#pragma once
#ifndef M_GLFW_UTILS
#define M_GLFW_UTILS
#include<GL/glew.h>
#include<GLFW/glfw3.h>

#include<imgui/imgui.h>
#include<imguizmo/ImGuizmo.h>
#include<imgui/backends/imgui_impl_glfw.h>
#include<imgui/backends/imgui_impl_opengl3.h>

#include<stb_image/stb_image.h>

#include<iostream>

#include<Eigen/Eigen>

#include<geometry/mesh.h>

#define LIGHT_SPHERE_SHADER "./data/shaders/light_sphere.shader"
#define BASIC_SHADER "./data/shaders/coordinate.shader"

#define GL_CHECK(x) do{ \
    GLClearError(); \
    x;  \
    if(GLCheckError()){ \
        std::cout << "[OPENGL ERROR] Error happen at " << #x << std::endl;  \
    }   \
}while(0)

static void GLClearError(){
    while(glGetError() != GL_NO_ERROR);
}

static bool GLCheckError(){
    while(GLenum error = glGetError()){
        std::cout << std::hex <<"[OpenGL Error] (" << error << ") Please check glew.h with error code! " << std::endl;
        return true;
    }
    return false;
}


class GLShader{
    private:
        uint32_t shader;
        struct ShaderProgramSource{
            std::string vert_source;
            std::string frag_source;
        };
    public:
        GLShader(std::string shader_path){ recompile(shader_path); }
        void create_shader(const std::string &vert_shader, const std::string &frag_shader){
            GL_CHECK(shader = glCreateProgram());
            uint32_t vs = compile_shader(GL_VERTEX_SHADER, vert_shader);
            uint32_t fs = compile_shader(GL_FRAGMENT_SHADER, frag_shader);
            
            GL_CHECK(glAttachShader(shader, vs));
            GL_CHECK(glAttachShader(shader, fs));
            GL_CHECK(glLinkProgram(shader));
            GL_CHECK(glValidateProgram(shader));

            GL_CHECK(glDeleteShader(vs));
            GL_CHECK(glDeleteShader(fs));
        }
        uint32_t compile_shader(uint32_t type, const std::string &source){
            uint32_t id;
            GL_CHECK(id = glCreateShader(type)); // create a shader and allocate a name 
            const char* src = source.c_str();   
            GL_CHECK(glShaderSource(id, 1, &src, nullptr));   // generate target src code
            GL_CHECK(glCompileShader(id));    // compile shader

            // TODO: error handling
            int result;
            GL_CHECK(glGetShaderiv(id, GL_COMPILE_STATUS, &result));
            if(result == GL_FALSE){
                int length;
                GL_CHECK(glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length));
                char *message = (char*)alloca(length * sizeof(char));
                GL_CHECK(glGetShaderInfoLog(id, length, &length, message));
                std::cout << "Failed to compile" << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") <<" shader!" << std::endl;
                std::cout << message << std::endl;
                GL_CHECK(glDeleteShader(id));
                return 0;
            }
            return id;
        }
        ShaderProgramSource ParseShader(const std::string &filepath){
            std::ifstream stream(filepath);

            enum class ShaderType{
                NONE = -1,
                VERTEX = 0,
                FRAGMENT = 1
            };

            std::string line;
            std::stringstream ss[2];
            ShaderType type = ShaderType::NONE;
            while(getline(stream, line)){
                if(line.find("#shader") != std::string::npos){
                    if(line.find("vertex") != std::string::npos){
                        //set mode to vertex
                        type = ShaderType::VERTEX;
                    }else if(line.find("fragment") != std::string::npos){
                        //set mode to fragment
                        type = ShaderType::FRAGMENT;
                    }
                }else{
                    if(type != ShaderType::NONE){
                        ss[(int)type] << line << "\n";
                    }
                }
            }

            return {ss[0].str(), ss[1].str()};
            stream.close();
        }

        void recompile(const std::string shader_path){
            ShaderProgramSource src = ParseShader(shader_path);
            create_shader(src.vert_source, src.frag_source);
            std::cout << "Compiled shader: " << shader_path << std::endl;
        }
        int bind_uniform(std::string uniform_name){
            int location = -1;
            GL_CHECK(location = glGetUniformLocation(shader, uniform_name.c_str()));
            if(location == -1)
                std::cout << "[GLShader Warning] Uniform: " << uniform_name << " doesn't exist!" << std::endl;
            return location;
        }
        void bind(){GL_CHECK(glUseProgram(shader));}
        void unbind(){GL_CHECK(glUseProgram(0));}
        void del(){GL_CHECK(glDeleteProgram(shader));}
        ~GLShader(){del();}

};

class Texture{
    private:
        uint32_t TB = 0;
        uint8_t *m_data;
        int m_width, m_height, m_channels;

    public:
        Texture(std::string path){load(path);}
        Texture(){}
        ~Texture(){
            if(TB)GL_CHECK(glDeleteTextures(1, &TB));
            
            if(m_data)free(m_data);
        }
        bool load(std::string path){
            m_data = stbi_load(path.c_str(), &m_width, &m_height, &m_channels, 0);
            image_fit_gl();
            GL_CHECK(glGenTextures(1, &TB));
            GL_CHECK(glBindTexture(GL_TEXTURE_2D, TB));
            GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, color_type(), m_width, m_height, 0, color_type(), GL_UNSIGNED_BYTE, (void*)m_data));
            GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP));
            GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP));
            GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
            std::cout << "[Load Texture] Shape : (" << m_width << ", " << m_height << ", " << m_channels << ")" << std::endl;
            return m_data != nullptr;
        }
        uint32_t color_type() const{return m_channels == 3 ? GL_RGB : GL_RGBA;}
        uint32_t width() const{return (uint32_t)m_width;}
        uint32_t height() const{return (uint32_t)m_height;}
        uint32_t channels() const{return (uint32_t)m_channels;}
        void bind(uint32_t slot = 0){
            GL_CHECK(glActiveTexture(GL_TEXTURE0 + slot));
            GL_CHECK(glBindTexture(GL_TEXTURE_2D, TB));
        }
        void unbind(){GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));}
        static void draw(Texture &texture){
            texture.bind();
            GL_CHECK(glEnable(GL_TEXTURE_2D));
            GL_CHECK(glBegin(GL_POLYGON));	//设置为多边形纹理贴图方式并开始贴图
            GL_CHECK(glVertex2f(-1.0f, -1.0f)); GL_CHECK(glTexCoord2f(0.0f, 0.0f)); 	//纹理左上角对应窗口左上角
            GL_CHECK(glVertex2f(-1.0f,  1.0f)); GL_CHECK(glTexCoord2f(0.0f, 1.0f)); 	//纹理左下角对应窗口左下角
            GL_CHECK(glVertex2f( 1.0f,  1.0f)); GL_CHECK(glTexCoord2f(1.0f, 1.0f)); 	//纹理右下角对应窗口右下角
            GL_CHECK(glVertex2f( 1.0f, -1.0f)); GL_CHECK(glTexCoord2f(1.0f, 0.0f)); 	//纹理右上角对应窗口右上角
            GL_CHECK(glEnd());	//结束贴图
            GL_CHECK(glDisable(GL_TEXTURE_2D));
            texture.unbind();
        }
    private:
        void image_fit_gl(){
            size_t width = m_width * m_channels;
            for(int i=0; i<m_height/2; i++){   // line
                for(int j=0; j<width; j++){   // value
                    size_t idx = j + i * width;
                    size_t targetIdx = j + (m_height - i - 1) * width;
                    uint8_t p = m_data[idx];
                    m_data[idx] = m_data[targetIdx];
                    m_data[targetIdx] = p;
                }
            }
        }
};


// Graphics Drawable API
// Should be call after glew init
class Drawable{
    protected:
        // buffer
        uint32_t VB = 0;
        uint32_t EB = 0;
        // shader
        GLShader shader;

    public:
        Drawable(std::string shader_path): shader(shader_path){}
        virtual void draw() = 0;
        void change_shader(std::string shader_path){shader.recompile(shader_path);}
};

#endif