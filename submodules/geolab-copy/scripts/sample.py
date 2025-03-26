import sys
import os
import numpy as np
WORK_SPACE = os.getcwd()
sys.path.insert(0, os.path.join(WORK_SPACE, 'build'))
import pygeo

def test_sample_on_triangles():
    triangles = pygeo.Mesh.load_triangles('./data/44234.obj')
    print(len(triangles))
    aabb = pygeo.BoundingBox()
    # aabb.min = np.asarray([-1.0, -1.0, -1.0])
    # aabb.max = np.asarray([1.0, 1.0, 1.0])
    weights = np.asarray([0.0, 3.0, 0.0]).reshape((3, 1))
    samples = pygeo.sample_on_triangles(triangles, 100000, weights)
    samples = np.asarray(samples)
    np.savetxt('sample.txt', samples)
    
if __name__ == '__main__':
    test_sample_on_triangles()