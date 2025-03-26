# GEOLAB
Project of geolab
## dependencies
    sudo apt-get install libglfw3-dev libxcursor-dev libxinerama-dev libxi-dev libglew-dev

## compiler version
    g++-9
## Build
    cmake . -B build
    cmake --build build 

    or

    bash run.sh

## Python Dependencies
    pip install imageio>=2.15
    pip install lpips

## Options Dependencies
    1. optix


## TODO
    1. Voxelization code
    2. marching cube

## update log
    
    VERSION 0.04 we use optix to accelerate ray tracing & SDF calculate