#pragma once
#ifndef MESH
#define MESH

#include<Eigen/Eigen>
#include<iostream>
#include<iomanip>
#include<vector>
#include<string>
#include<sstream>
#include<fstream>
#include<unordered_map>

#include<utils/common.cuh>
#include<geometry/triangle.cuh>

#include<io/obj_loader.h>
#include<io/stl_reader.h>
#include<tinyply.h>

// Used to render model
class Mesh{
    public:
        struct Vertex{
            Eigen::Vector3f position;
            Eigen::Vector3f normal;
            Eigen::Vector2f textureCoord;

            Vertex(){}
            Vertex(Eigen::Vector3f p, Eigen::Vector3f n, Eigen::Vector2f t): position(p), normal(n), textureCoord(t){}
        };
        typedef Eigen::Vector<uint32_t, 3> Face3;

    private:
        std::vector<Mesh::Vertex> vertices;
        std::vector<Mesh::Face3> faces;

    public:
        Mesh(){}
        Mesh(std::string path, bool do_normalize = false){
            load_mesh(path, vertices, faces);
            if(do_normalize)normalize(vertices);
        }
        ~Mesh(){}
        void load(std::string path){load_mesh(path, vertices, faces);}
        void load(const std::vector<Triangle> &triangles){
            vertices.clear();
            faces.clear();

            for(auto tri : triangles){
                vertices.emplace_back(tri.a, tri.normal(), Eigen::Vector2f::Zero());
                vertices.emplace_back(tri.b, tri.normal(), Eigen::Vector2f::Zero());
                vertices.emplace_back(tri.c, tri.normal(), Eigen::Vector2f::Zero());
                faces.emplace_back(vertices.size()-3, vertices.size()-2, vertices.size()-1);
            }
        }
        void load(const std::vector<Eigen::Vector3f> &vertices_, const std::vector<Mesh::Face3> &faces_){
            vertices.resize(vertices_.size());
            faces.resize(faces_.size());

            for(uint32_t i=0; i<vertices_.size(); i++){
                vertices[i].position = vertices_[i];
            }

            for(uint32_t i=0; i<faces_.size(); i++){
                auto f = faces_[i];
                auto &a = vertices[f.x()];
                auto &b = vertices[f.y()];
                auto &c = vertices[f.z()];
                Eigen::Vector3f normal = (b.position - a.position).cross(c.position - a.position).normalized();
                a.normal = normal;
                b.normal = normal;
                c.normal = normal;
                faces[i] = f;
            }
        }
        void load(const std::vector<Eigen::Vector3f> &vertices_, const std::vector<uint32_t> &faces_){
            if(faces_.size() % 3 != 0)return;
            vertices.resize(vertices_.size());
            faces.resize(faces_.size());

            for(uint32_t i=0; i<vertices_.size(); i++){
                vertices[i].position = vertices_[i];
            }

            for(uint32_t i=0; i<faces_.size(); i+=3){
                auto fa = faces_[i];
                auto fb = faces_[i+1];
                auto fc = faces_[i+2];
                auto &a = vertices[fa];
                auto &b = vertices[fb];
                auto &c = vertices[fc];
                Eigen::Vector3f normal = (b.position - a.position).cross(c.position - a.position).normalized();
                a.normal = normal;
                b.normal = normal;
                c.normal = normal;
                faces[i/3] = {fa, fb, fc};
            }

        }
        void load(const std::vector<Eigen::Vector3f> &vertices_, const std::vector<Eigen::Vector3f> &normals_, const std::vector<Mesh::Face3> &faces_){
            assert(vertices_.size() == normals_.size());
            vertices.resize(vertices_.size());
            faces.resize(faces_.size());

            for(uint32_t i=0; i<vertices_.size(); i++){
                vertices[i].position = vertices_[i];
                vertices[i].normal = normals_[i];
            }

            for(uint32_t i=0; i<faces_.size(); i++){
                faces[i] = faces_[i];
            }
        }

        Mesh::Vertex* data_v() { return vertices.data(); }
        Mesh::Face3*  data_f() { return faces.data(); }
        size_t  size_v() const{ return vertices.size(); }
        size_t  size_f() const{ return faces.size(); }
        size_t  bytes_v() const{ return vertices.size() * sizeof(Mesh::Vertex); }
        size_t  bytes_f() const{ return faces.size() * sizeof(Mesh::Face3); }

        std::vector<Triangle> format_triangles(){
            std::vector<Triangle> triangles;
            for(auto f : faces){
                auto a = vertices[f.x()];
                auto b = vertices[f.y()];
                auto c = vertices[f.z()];
                triangles.emplace_back(a.position, b.position, c.position);
            }
            return triangles;
        }
        std::vector<Eigen::Matrix3f> format_triangles_matrix(){
            std::vector<Eigen::Matrix3f> triangles;
            for(auto f : faces){
                Eigen::Matrix3f tri;
                auto a = vertices[f.x()].position;
                auto b = vertices[f.y()].position;
                auto c = vertices[f.z()].position;
                tri.col(0) = a;
                tri.col(1) = b;
                tri.col(2) = c;
                triangles.emplace_back(tri);
            }
            return triangles;
        }


        static std::unique_ptr<Mesh> create(){ return std::unique_ptr<Mesh>(new Mesh()); }
        static int load_mesh(std::string path, std::vector<Vertex> &vertices, std::vector<Face3> &faces){
            std::string suffix = path.substr(path.size() - 4, 4);
            vertices.clear();
            faces.clear();
            if(suffix == ".obj"){
                objl::Loader loader;
                bool loadout = loader.LoadFile(path);
                if(!loadout)return 0;
                std::vector<objl::Vertex> vs = loader.LoadedMeshes[0].Vertices;
                std::vector<unsigned int> is = loader.LoadedMeshes[0].Indices;

                for(auto v : vs){
                    Vertex nv;
                    nv.position = {v.Position.X, v.Position.Y, v.Position.Z};
                    nv.normal = {v.Normal.X, v.Normal.Y, v.Normal.Z};
                    nv.textureCoord = {v.TextureCoordinate.X, v.TextureCoordinate.Y};
                }

                for(int i=0; i<is.size(); i+=3){
                    faces.push_back({is[i], is[i+1], is[i+2]});
                }

            }else if(suffix == ".stl"){
                std::vector<STLTriangle> stl_triangles;
                load_stl(path, stl_triangles, nullptr);

                for(auto st : stl_triangles){
                    vertices.push_back({st.vertices[0], st.normal, Eigen::Vector2f::Zero()});
                    vertices.push_back({st.vertices[1], st.normal, Eigen::Vector2f::Zero()});
                    vertices.push_back({st.vertices[2], st.normal, Eigen::Vector2f::Zero()});
                    Face3 f = {(uint32_t)vertices.size()-3, (uint32_t)vertices.size()-2, (uint32_t)vertices.size()-1};
                    faces.push_back(f);
                }
            }else if(suffix == ".ply"){
                std::cout << "ply format is not supported now" << std::endl;
                return 0;
            }else{
                std::cout << suffix << " format is not available" << std::endl;
                return -1;
            }
            return 1;
        }
        static int load_triangles(std::string path, std::vector<Triangle> &triangles){
            std::string suffix = path.substr(path.size() - 4, 4);
            triangles.clear();
            if(suffix == ".obj"){
                objl::Loader loader;
                bool loadout = loader.LoadFile(path);
                if(!loadout)return 0;
                std::vector<objl::Vertex> vertices = loader.LoadedMeshes[0].Vertices;
                std::vector<unsigned int> indices = loader.LoadedMeshes[0].Indices;

                for(int i=0; i<indices.size(); i+=3){
                    Triangle tri;
                    tri.a = {vertices[indices[i+0]].Position.X, vertices[indices[i+0]].Position.Y, vertices[indices[i+0]].Position.Z};
                    tri.b = {vertices[indices[i+1]].Position.X, vertices[indices[i+1]].Position.Y, vertices[indices[i+1]].Position.Z};
                    tri.c = {vertices[indices[i+2]].Position.X, vertices[indices[i+2]].Position.Y, vertices[indices[i+2]].Position.Z};
                    triangles.push_back(tri);
                }
            }else if(suffix == ".stl"){
                std::vector<STLTriangle> stl_triangles;
                load_stl(path, stl_triangles, nullptr);

                for(auto st : stl_triangles){
                    Triangle tri;
                    tri.a = st.vertices[0];
                    tri.b = st.vertices[1];
                    tri.c = st.vertices[2];
                    if(tri.normal().dot(st.normal) < 0){
                        auto temp = tri.b;
                        tri.b = tri.c;
                        tri.c = temp;
                    }
                    triangles.push_back(tri);
                }
            }else if(suffix == ".ply"){
                std::cout << "ply format is not supported now" << std::endl;
                return 0;
            }else{
                std::cout << suffix << " format is not available" << std::endl;
                return -1;
            }
            return 1;
        }
        static std::vector<Triangle> load_triangles(std::string path){
            std::vector<Triangle> triangles;
            load_triangles(path, triangles);
            return triangles;
        }
        
        /**
         * mode & 1 normal available
         * mode & 2 texcoord available
         */
        static void save_obj(std::string path, std::vector<Vertex> &vertices, std::vector<unsigned int> &faces, char mode = 1){
            std::ofstream os(path);
            os << std::setprecision(4);
            for(Vertex v : vertices){
                os << "v " << v.position.x() << " " << v.position.y() << " " << v.position.z() << "\n";
                if(mode & 1u){
                    os << "vn " << v.normal.x() << " " << v.normal.y() << " " << v.normal.z() << "\n";
                }
                if(mode & 2u){
                    os << "vt " << v.textureCoord.x() << " " << v.textureCoord.y() << "\n";
                }
                    
            }
            for(uint32_t i=0; i<faces.size(); i+=3){
                os << "f";
                for(uint32_t j=0; j<3; j++){
                    os << " " << (faces[i+j] + 1);
                    if(mode & 3u){
                        os << "/";
                        if(mode & 2u){
                            os << (faces[i+j] + 1);
                        }
                        if(mode & 1u){
                            os << "/" << (faces[i+j] + 1);
                        }
                    }
                }
                os << "\n";
            }
            os.close();
        }
        static void normalize(std::vector<Vertex> &vertices){
            Eigen::Vector3f maxb = Eigen::Vector3f::Constant(-50.0f);
            Eigen::Vector3f minb = Eigen::Vector3f::Constant(50.0f);
            Eigen::Vector3f center = Eigen::Vector3f::Zero();
            for(int i=0; i<vertices.size(); i++){
                Vertex &vertex = vertices[i];
                maxb = maxb.cwiseMax(vertex.position);
                minb = minb.cwiseMin(vertex.position);
            }
            center = (maxb + minb) / 2;
            float scale = 0.0f;
            for(auto &v : vertices){
                v.position -= center;
                scale = std::max(scale, v.position.norm());
            }
            scale = 1.0f / scale;
            for(auto &v: vertices){
                v.position *= scale;
            }
        }
        
};
#endif