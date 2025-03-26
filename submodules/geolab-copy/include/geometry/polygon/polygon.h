#pragma once
#ifndef POLYGON_H
#define POLYGON_H

#include<geometry/mesh.h>
#include<Eigen/Eigen>
#include<vector>
#include<string>
#include<sstream>

template<uint32_t STRIDE>
class Polygon{
    public:
        typedef Eigen::Vector3f Vert;
        typedef Eigen::Vector<uint32_t, 2> Edge;
        typedef Eigen::Vector<uint32_t, STRIDE> Face;

    protected:
        Polygon(){}
        virtual std::unique_ptr<Mesh> triangulate() = 0; 

    protected:
        std::vector<Polygon::Vert> V;
        std::vector<Polygon::Edge> E;
        std::vector<Polygon::Face> F;

};




#endif