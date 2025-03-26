import sys
import os
import numpy as np
import pandas as pd
WORK_SPACE = os.getcwd()
sys.path.insert(0, os.path.join(WORK_SPACE, 'build'))
import pygeo
import imageio

def test_sdf_renderer():

    
    camera_pos = pygeo.sample_fibonacci(50)
    camera_pos = np.asarray(camera_pos) * 2.8 * 1.71

    frame = np.load('./data/frame_26.npz')
    win_res = [frame['arr_0'].shape[0], frame['arr_0'].shape[1]]
    rows = win_res[0] * win_res[1]
                        
    renderer = pygeo.SDFRenderer(win_res, True)
    renderer.render_ground_truth = False;
    
    points = frame['arr_0'].reshape((rows, 3))
    normals = frame['arr_1'].reshape((rows, 3))
    hits = frame['arr_2'].reshape((rows, 1))
    n_steps = frame['arr_3'].reshape((rows, 1))
    dis = frame['arr_4'].reshape((rows, 1))

    picture = renderer.read_and_render_frame(points, normals, hits, n_steps, dis, camera_pos[26])
    picture = np.asarray(picture).reshape((win_res[0], win_res[1], 4))
    picture = np.asarray(picture*255, dtype=np.uint8)
    imageio.imwrite('./test.png', picture)
    
    return 0

if __name__ == '__main__':
    test_sdf_renderer()
