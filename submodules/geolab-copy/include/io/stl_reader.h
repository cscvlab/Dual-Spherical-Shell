#pragma once
#ifndef STL_READER
#define STL_READER

#include<Eigen/Eigen>
#include<iostream>
#include<fstream>
#include<vector>
#include<string>

#include<utils/common.cuh>

struct STLHead{
    char partName[80];
    int faceNum;
};

struct STLTriangle{
    Eigen::Vector3f normal;
    Eigen::Vector3f vertices[3];
    char info[2];
};

enum STLFileType{
    BINARY,
    ASCII
};

inline int load_stl(std::string path, std::vector<STLTriangle> &triangles, STLHead *head = nullptr, STLFileType type = STLFileType::BINARY){
    if( path.substr(path.size() - 4, 4) != ".stl")return -1;
    
    if(type == STLFileType::BINARY){
        std::ifstream file(path, std::ios::in | std::ios::binary);
        if(!file.is_open())return 0;
        STLTriangle triangle;
        triangles.clear();

        if(head){
            file.read((char*)head, sizeof(STLHead));
        }else{
            STLHead tmp_head;
            file.read((char*)&tmp_head, sizeof(STLHead));
        }
        while(file.read((char*)&triangle, sizeof(STLTriangle))){
            triangles.push_back(triangle);
        }
        std::cout << "Load Triangles: " << triangles.size() << std::endl;
        file.close();
        return 1;
    }else{
        std::ifstream file(path);
        if(!file.is_open())return 0;
        std::string line;
        std::vector<std::string> words;
        STLTriangle triangle;

        getline(file, line);    // first line
        split(line, words, ' ');
        if(words[0] == "solid"){
            int i = 0;
            while(getline(file, line)){ // facet normal
                split(line, words, ' ');
                if(words[0] == "facet"){
                    triangle.normal = Eigen::Vector3f(
                        std::stof(words[2]), 
                        std::stof(words[3]), 
                        std::stof(words[4])
                    );
                }else if(words[0] == "vertex"){
                    triangle.vertices[i++] = Eigen::Vector3f(
                        std::stof(words[1]), 
                        std::stof(words[2]), 
                        std::stof(words[3])
                    );
                }else if(words[0] == "endfacet"){   // push and reset
                    i = 0;
                    triangles.push_back(triangle);
                }
            }
        }   // end of reading
        std::cout << "Load Triangles: " << triangles.size() << std::endl;
        file.close();
        return 1;
    }
}

#endif