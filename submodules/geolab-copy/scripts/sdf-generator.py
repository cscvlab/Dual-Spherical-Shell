import sys
import os
import numpy as np
WORK_SPACE = os.getcwd()
sys.path.insert(0, os.path.join(WORK_SPACE, 'build'))
import pygeo


def test_sdf_generator():
    gen = pygeo.SDFGenerator()
    gen.load_obj('./data/91120.obj')

    AxisSize = 128
    pts = np.array([(i, j, k) for i in range(AxisSize) for j in range(AxisSize) for k in range(AxisSize)], dtype=np.float32)
    pts = (pts - AxisSize/2) / (AxisSize / 2)
    # pos = pts
    # print(pos.shape, pos.dtype)
    # pts = (np.random.rand(262114, 3) - 0.5) * 2
    pts = np.asarray(pts, dtype=np.float32)

    distance = gen.generate_sdf(pts, pygeo.SDFCalcMode.RAYSTAB)
    # print(distance)
    distance = np.asarray(distance)
    print(distance.shape)
    mask = np.where(distance <= 0)
    o_mask = np.where(distance > 0)
    # print(pts[mask])
    np.savetxt('inner.txt', pts[mask])
    np.savetxt('outer.txt', pts[o_mask])

if __name__ == '__main__':
    test_sdf_generator()