import random

import numpy as np


class Supshape:
    """
    Implementation for box (Cuboid) RANSAC.

    A cuboid is defined as convex polyhedron bounded by six faces formed by three orthogonal normal vectors. Cats love to play with this kind of geometry.
    This method uses 6 points to find 3 best plane equations orthogonal to eachother.

    We could use a recursive planar RANSAC, but it would use 9 points instead. Orthogonality makes this algorithm more efficient.

    ![Cuboid](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/cuboid.gif "Cuboid")

    ---
    """

    def __init__(self):
        self.inliers = []
        self.equation = []

    def fit(self, pts, AA,BB, thresh=0.05, maxIteration=10000000):
        n_points = pts.shape[0]
        best_eq = []
        bav=0
        best_inliers = []
        bestplane=0
        it=maxIteration
        while it>0:
            plane_eq = []
            print(it)
            # Samples 6 random points
            id_samples = random.sample(range(0, n_points), 2)
            pt_samples = pts[id_samples]
            vecA_norm = AA
            vecB_norm =pt_samples[0, :] - BB
            vecC_norm = pt_samples[1, :] - BB
            st1=np.stack([vecA_norm] * 2, 0)
            dist = np.cross(st1, (BB - [pt_samples[0, :],pt_samples[1, :]]))
            dist=np.linalg.norm(dist, axis=1)
            # st2 = np.stack([vecA_norm] * 1, 0)
            # dist2 = np.cross(st2, (pt_samples[2, :] - [ pt_samples[3, :]]))
            # dist2 = np.linalg.norm(dist2, axis=1)
            # print(it,dist)
            if    dist[0]<3or dist[1]<3or abs(dist[0]-dist[1])>1:
                continue

            av=(dist[0]+dist[1])/2
            it -= 1



            # Distance from a point to a line
            pt_id_inliers = []  # list of inliers ids
            vecC_stakado = np.stack([vecA_norm] * n_points, 0)
            dist_pt = np.cross(vecC_stakado, (pt_samples[0, :] - pts))
            dist_pt = np.linalg.norm(dist_pt, axis=1)
            dist_pt2=np.cross(vecC_stakado, (pt_samples[1, :] - pts))
            dist_pt2 = np.linalg.norm(dist_pt2, axis=1)
            pt_id_inliers = np.where((np.abs(dist_pt) <= thresh) | (np.abs(dist_pt2) <= thresh))[0]
            if len(pt_id_inliers)> len(best_inliers):
                    bav=av
                    best_inliers = pt_id_inliers
                    self.inliers = best_inliers
                    self.A = vecA_norm
                    self.B = pt_samples[0, :]
                    self.C=pt_samples[1, :]

        return self.A, self.B, self.inliers,self.C
