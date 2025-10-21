from scipy.interpolate import splprep, splev
import numpy as np
import vg
from pytransform3d.rotations import matrix_from_axis_angle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as R



_points = np.array(
        [[0, 0, 0],
         [0, 1, 0],
         [1, 1, 0],
         [1, 0, 0]],
    ) - np.array([0.5, 0.5, 0])
_normal = np.array([0, 2, 2])

def ee( x, y, z, n=3, degree=2, **kwargs):
    assert n >= 3
    tck, u = splprep([x, y, z], s=0, k=2)
    evalpts = np.linspace(0, 1, n)
    pts = np.array(splev(evalpts, tck))
    der = np.array(splev(evalpts, tck, der=1))
    points = []
    for i in range(n):
        points_slice = rotmat(der[:, i], _points)
        points_slice = points_slice + pts[:, i]
        points.append(points_slice)

    points = np.stack(points)
    return points


def rotmat(vector, points):
    """
    Rotates a 3xn array of 3D coordinates from the +z normal to an
    arbitrary new normal vector.
    """

    print(vector)
    vector = vg.normalize(vector)
    axis = vg.perpendicular(vg.basis.z, vector)
    angle = vg.angle(vg.basis.z, vector, units='rad')

    a = np.hstack((axis, (angle,)))
    R = matrix_from_axis_angle(a)

    r = Rot.from_matrix(R)
    rotmat = r.apply(points)
    r2=np.dot(points,R.T)
    print(R,rotmat)
    return rotmat,r2
x = [0, 1, 2, 3, 6]
y = [0, 2, 5, 6, 2]
z = [0, 0, 0, 0, 0]

point=np.array(
        [[0, 0, 0],
         [1, 2, 0],
         [2, 5, 0],
         [3, 6, 0],
         [6, 2, 0]],
    )

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(x, y, z)
planes,p2 = rotmat(_normal,point)
ax.scatter(planes[:,  0], planes[:,  1], planes[:,  2])
ax.scatter(p2[:,  0], p2[:,  1], p2[:,  2])
plt.show()