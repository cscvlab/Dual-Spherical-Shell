import numpy as np
import sys
import os
WORK_SPACE = os.getcwd()
sys.path.insert(0, os.path.join(WORK_SPACE, 'build'))
import pygeo

def voxelize():
    obj_path = './data/44234.obj'
    triangles = pygeo.Mesh.load_triangles(obj_path)
    grid = pygeo.DualVoxel([256, 256, 256])
    surface_num = grid.voxelize_triangles(triangles)
    print(surface_num)
    inside_num = grid.fill_inside()
    print(inside_num)
    surface = np.asarray(grid.surface_voxel_center()).reshape((surface_num, 3))
    inside = np.asarray(grid.inside_voxel_center()).reshape((inside_num, 3))
    np.savetxt('surface.txt',surface)
    np.savetxt('inside.txt', inside)

def voxelize_dynamic():
    obj_path = './data/44234.obj'
    grid_loader = pygeo.DualVoxelDynamic(pygeo.SDFCalcMode.RAYSTAB)
    grid_loader.load_model(obj_path)
    surface = np.asarray(grid_loader.surface_voxel_center([64, 64, 64]))
    # surface = np.asarray(pygeo.sample_voxel_center([64, 64, 64]))
    print(surface.shape)
    # inside = np.asarray(grid_loader.inside_voxel_center([256, 256, 256]))
    # print(inside.shape)
    np.savetxt('surfaced.txt', surface)
    # np.savetxt('insided.txt', inside)

if __name__ == '__main__':
    voxelize()