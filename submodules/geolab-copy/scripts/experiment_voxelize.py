import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
WORK_SPACE = os.getcwd()
sys.path.insert(0, os.path.join(WORK_SPACE, 'build'))
import pygeo

coordinate_types = ['INTEGER', 'FLOAT']
output_types = ['txt', 'npz']

obj_path = '/media/cscvlab/DATA/datasets/public_dataset/shapenet150'
output_path = '/media/cscvlab/DATA/datasets/public_dataset/shapenet150_voxel_center_256'

coordinate_type = coordinate_types[1]
output_type = output_types[0]
resolution = [256, 256, 256]
seperate = ','
fmt = '%4d' if coordinate_type == 'INTEGER' else '%.6f'

def voxelize():
    obj_files = os.listdir(obj_path)
    task = tqdm(obj_files)
    for obj_file in task:
        words = obj_file.split('.')
        if words[1] != 'obj':
            continue
        
        path = os.path.join(obj_path, obj_file)
        triangles = pygeo.Mesh.load_triangles(path)
        grid = pygeo.DualVoxel(resolution)
        surface_num = grid.voxelize_triangles(triangles)
        inside_num = grid.fill_inside()
        task.set_description('obj: {}, surface: {}, inside: {}'.format(words[0], surface_num, inside_num))
        
        if coordinate_type == 'FLOAT':
            surface = np.asarray(grid.surface_voxel_center(), dtype=np.float32)
            inside = np.asarray(grid.inside_voxel_center(), dtype=np.float32)
            outside = np.asarray(grid.outside_voxel_center(), dtype=np.float32)
        else:
            surface = np.asarray(grid.surface_voxel(), dtype=np.int32)
            inside = np.asarray(grid.inside_voxel(), dtype=np.int32)
            outside = np.asarray(grid.outside_voxel(), dtype=np.int32)
            
        if output_type == 'txt':
            output = os.path.join(output_path, words[0])
            
            if not os.path.exists(output):
                os.makedirs(output)
            
            np.savetxt(output + '/surface.txt', surface, fmt=fmt)
            np.savetxt(output + '/inside.txt', inside, fmt=fmt)
            np.savetxt(output + '/outside.txt', outside, fmt=fmt)
            
        elif output_type == 'npz':
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                
            np.savez_compressed(output_path + '/{}.npz'.format(words[0]), surface=surface, inside=inside, outside=outside)
            

if __name__ == '__main__':
    voxelize()