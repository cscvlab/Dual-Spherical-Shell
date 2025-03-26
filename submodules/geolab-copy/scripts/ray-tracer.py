import sys
import os
import numpy as np
import pandas as pd
WORK_SPACE = os.getcwd()
sys.path.insert(0, os.path.join(WORK_SPACE, 'build'))
import pygeo
import imageio

obj_path = './data/44234.obj'
resolution = [1024, 1024]

def ray_tracer_window():
    
    camera_pos = pygeo.sample_fibonacci(50)
    camera_pos = np.asarray(camera_pos) * 2.8 * 1.73205080756887729
    
    renderer = pygeo.SDFRenderer(resolution)
    renderer.light.pos = [1.0, 1.0, 1.0]
    renderer.draw_light_sphere = True
    renderer.draw_coordinate_axis = True
    
    renderer.light.background_color = [0.0, 0.0, 0.0, 0.0]
    renderer.build_bvh('./data/53750.obj')
    # renderer.scene.slice_plane_z = 1.0
    # renderer.scene.surfaceColor = [1.0, 1.0, 1.0]
    # renderer.light.specular = 0.3
    # renderer.light.kd = 0.8
    renderer.render_ray_trace()


def test_ray_tracer():
    renderer = pygeo.SDFRenderer()

    # Light & Scene config
    renderer.scene.slice_plane_z = 1.0
    renderer.scene.surfaceColor = [1.0, 1.0, 1.0]
    renderer.light.specular = 0.3
    renderer.light.kd = 0.8

    camera_pos = pygeo.sample_fibonacci(50)
    camera_pos = np.asarray(camera_pos) * 2.8 * 1.73205080756887729
    
    pictures = renderer.render_ray_trace('./data/53750.obj', resolution, pygeo.ERenderMode.POSITION, camera_pos)
    pictures = np.asarray(pictures).reshape((camera_pos.shape[0], resolution[0], resolution[1], 4))
    pictures = np.asarray(pictures*255, dtype=np.uint8)
    for i in range(camera_pos.shape[0]):
        imageio.imwrite('./pictures/gltest_{}.png'.format(i), pictures[i])

def ray_tracer_utils_test():
    rtu = pygeo.RayTracerUtils(obj_path)
    positions = np.asarray([[x/128.0, y/128.0, -1.0] for x in range(-64, 64) for y in range(-64, 64)])
    directions = np.asarray([[0.0, 0.0, 1.0] for i in range(128) for j in range(128)])
    print(positions.shape)
    print(directions.shape)
    hit = np.asarray(rtu.trace(positions, directions))
    print(hit[:,2] < 1.0)
    hit = hit[(hit[:,2] < 1.0)]
    np.savetxt('positions.txt', positions)
    np.savetxt('hit.txt', hit)

if __name__ == '__main__':
    # test_ray_tracer()
    # ray_tracer_window()
    ray_tracer_utils_test()