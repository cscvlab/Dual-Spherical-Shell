#pragma once
#ifndef POLYGON_CUBE_H
#define POLYGON_CUBE_H

#include<geometry/polygon/polygon.h>

class Cube : public Polygon<4>{
    public:
        Cube(
            Eigen::Vector3f min = Eigen::Vector3f::Constant(-1.0f), 
            Eigen::Vector3f max = Eigen::Vector3f::Ones()
        ){
            m_min = min;
            m_max = max;

            V.emplace_back(min.x(), min.y(), min.z());
            V.emplace_back(max.x(), min.y(), min.z());
            V.emplace_back(max.x(), max.y(), min.z());
            V.emplace_back(min.x(), max.y(), min.z());
            V.emplace_back(min.x(), min.y(), max.z());
            V.emplace_back(max.x(), min.y(), max.z());
            V.emplace_back(max.x(), max.y(), max.z());
            V.emplace_back(min.x(), max.y(), max.z());

            E.emplace_back(0u, 1u);
            E.emplace_back(1u, 2u);
            E.emplace_back(2u, 3u);
            E.emplace_back(3u, 0u);
            E.emplace_back(4u, 5u);
            E.emplace_back(5u, 6u);
            E.emplace_back(6u, 7u);
            E.emplace_back(7u, 8u);
            E.emplace_back(0u, 4u);
            E.emplace_back(1u, 5u);
            E.emplace_back(2u, 6u);
            E.emplace_back(3u, 7u);

            uint32_t faces[] = {
                0, 1, 2, 3, // front
                4, 5, 6, 7, // back
                0, 4, 7, 3, // left
                1, 5, 6, 2, // right
                0, 1, 5, 4, // bottom
                3, 2, 6, 7  // top
            };
            for(uint32_t i=0; i<32u; i++){
                F.emplace_back(faces[i]);
            }
        }
    
        ~Cube(){}

        std::unique_ptr<Mesh> triangulate() override{
            std::unique_ptr<Mesh> mesh = Mesh::create();
            return mesh;
        }
    private:
        Eigen::Vector3f m_min;
        Eigen::Vector3f m_max;
};

#endif