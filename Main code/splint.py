
import numpy as np
import matplotlib.pyplot as plt

# The NURBS Book Ex9.1
curve=np.asarray([[1,1,1],[2,2,2],[3,3,3]])
cp_curve=curve.copy()
print(cp_curve)
cp_curve[0]=cp_curve[-1]
print(cp_curve)
cp_curve=np.roll(cp_curve,-1,axis=0)
print(cp_curve)
lenth=(curve[:,0]-cp_curve[:,0])**2+(curve[:,1]-cp_curve[:,1])**2+(curve[:,2]-cp_curve[:,2])**2
print(np.sqrt(lenth))
# points = ((0, 0), (3, 4), (-1, 4), (-4, 0),(1,-1), (-4, -3),(1,-4))
# degree = 3  # cubic curve
#
# # Do global curve interpolation
# curve = fitting.interpolate_curve(points, degree)
# print(curve.knotvector)
#
# # Prepare points
# evalpts = np.array(curve.evalpts)
# print(len(evalpts))
# pts = np.array(points)
#
# # Plot points together on the same graph
# fig = plt.figure(figsize=(10, 8), dpi=96)
# plt.plot(evalpts[:, 0], evalpts[:, 1])
# plt.scatter(pts[:, 0], pts[:, 1], color="red")
# plt.show()
# import numpy as np
# from scipy import spatial
#
# xy1 = np.array(
#     [[243,  3173,1],
#      [525,  2997,33]])
#
# xy2 = np.array(
#     [[682, 2644,23],
#      [277, 2651,33],
#      [396, 2640,33]])
#
# # This solution is optimal when xy2 is very large
# tree = spatial.cKDTree(xy2)
# mindist, minid = tree.query(xy1)
# print(mindist)
#
# # This solution by @denis is OK for small xy2
# mindist = np.min(spatial.distance.cdist(xy1, xy2), axis=1)
# print(mindist)