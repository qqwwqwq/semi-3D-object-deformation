import random

import numpy as np


class width:
    """
    Implementation for 3D Line RANSAC.

    This object finds the equation of a line in 3D space using RANSAC method.
    This method uses 2 points from 3D space and computes a line. The selected candidate will be the line with more inliers inside the radius theshold.

    ![3D line](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/line.gif "3D line")

    ---
    """

    def __init__(self):
        self.inliers = []
        self.A = []
        self.B = []

    def fit(self, pts,T, thresh=0.2):
        """
        Find the best equation for the 3D line. The line in a 3d enviroment is defined as y = Ax+B, but A and B are vectors intead of scalars.

        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the line which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `A`: 3D slope of the line (angle) `np.array (1, 3)`
        - `B`: Axis interception as `np.array (1, 3)`
        - `inliers`: Inlier's index from the original point cloud. `np.array (1, M)`
        ---
        """
        n_points = pts.shape[0]
        pt_id_inliers = []
        dist_pt = (
                          T[0] * pts[:, 0] + T[1] * pts[:, 1] + T[2] * pts[:, 2] + T[3]
                  ) / np.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2)
        pt_id_inliers = np.where(np.abs(dist_pt) >= thresh)[0]
        best_inliers = pt_id_inliers
        self.inliers = best_inliers

        return self.inliers