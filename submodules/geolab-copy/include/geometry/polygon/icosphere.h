#pragma once
#ifndef POLYGON_ICOSPHERE_H
#define POLYGON_ICOSPHERE_H

#include<geometry/polygon/polygon.h>

class IcoSphere : public Polygon<3>{
    public:
        IcoSphere(uint32_t level = 3): m_level(level){
            // base
            float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
            V.emplace_back(Eigen::Vector3f(-1,  t,  0).normalized());
            V.emplace_back(Eigen::Vector3f( 1,  t,  0).normalized());
            V.emplace_back(Eigen::Vector3f(-1, -t,  0).normalized());
            V.emplace_back(Eigen::Vector3f( 1, -t,  0).normalized());
            V.emplace_back(Eigen::Vector3f( 0, -1,  t).normalized());
            V.emplace_back(Eigen::Vector3f( 0,  1,  t).normalized());
            V.emplace_back(Eigen::Vector3f( 0, -1, -t).normalized());
            V.emplace_back(Eigen::Vector3f( 0,  1, -t).normalized());
            V.emplace_back(Eigen::Vector3f( t,  0, -1).normalized());
            V.emplace_back(Eigen::Vector3f( t,  0,  1).normalized());
            V.emplace_back(Eigen::Vector3f(-t,  0, -1).normalized());
            V.emplace_back(Eigen::Vector3f(-t,  0,  1).normalized());

            F.emplace_back( 0, 11,  5);
            F.emplace_back( 0,  5,  1);
            F.emplace_back( 0,  1,  7);
            F.emplace_back( 0,  7, 10);
            F.emplace_back( 0, 10, 11);

            F.emplace_back( 1,  5,  9);
            F.emplace_back( 5, 11,  4);
            F.emplace_back(11, 10,  2);
            F.emplace_back(10,  7,  6);
            F.emplace_back( 7,  1,  8);

            F.emplace_back( 3,  9,  4);
            F.emplace_back( 3,  4,  2);
            F.emplace_back( 3,  2,  6);
            F.emplace_back( 3,  6,  8);
            F.emplace_back( 3,  8,  9);

            F.emplace_back( 4,  9,  5);
            F.emplace_back( 2,  4, 11);
            F.emplace_back( 0,  1,  7);
            F.emplace_back( 8,  6,  7);
            F.emplace_back( 9,  8,  1);


            // refine
            for(uint32_t l = 0; l < level; l++){
                std::vector<Polygon<3>::Face> new_faces;
                for(int j=0; j<F.size(); j++){
                    auto f = F[j];
                    uint32_t a = add_mid_point(f.x(), f.y());
                    uint32_t b = add_mid_point(f.y(), f.z());
                    uint32_t c = add_mid_point(f.z(), f.x());

                    new_faces.push_back({f.x(), a, c});
                    new_faces.push_back({f.y(), b, a});
                    new_faces.push_back({f.z(), c, b});
                    new_faces.push_back({    a, b, c});
                }
                F.clear();
                for(auto nf : new_faces){
                    F.push_back(nf);
                }
            }

        }
        ~IcoSphere(){}
        std::unique_ptr<Mesh> triangulate() override{
            auto mesh = Mesh::create();
            mesh->load(V, V, F);
            return mesh;
        }
        uint32_t level() const { return m_level; }
    private:
        uint32_t m_level;
        Eigen::Vector3f m_center;
        float m_radius;

        uint32_t add_mid_point(uint32_t a, uint32_t b){
            auto A = V[a], B = V[b];
            auto M = ((A + B) / 2).normalized();
            V.emplace_back(M);
            return V.size()-1;
        }
};

#endif