import random
import vg
import numpy as np
from cubic import cubicSplineInterpolate
from scipy import spatial
class ransacspline:
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
        self.supoint=[]
    def fit(self, pts, thresh, maxIteration=3000):
        n_points = pts.shape[0]
        best_inliers = []
        best=0
        it=maxIteration
        first = 1
        flag=0
        while it>0 or flag==0:
            print(it)
            it -= 1
            if first == 1:
            # Samples 6 random points
                id_samples = np.array(random.sample(range(0, n_points),10))
            # id_samples.sort()
                pt_samples = pts[id_samples]
                pt_samples=np.unique(pt_samples,axis=0)
            else:
                id_one = self.bsample.tolist()
                del id_one[random.randint(0, len(id_one) - 1)]
                del id_one[random.randint(0, len(id_one) - 1)]
                id_two =np.array(random.sample(range(0, n_points),2)).tolist()
                id_one.insert(random.randint(0, len(id_one)), id_two[0])
                id_one.insert(random.randint(0, len(id_one)), id_two[1])
                id_samples = np.array(id_one)
                pt_samples = pts[id_samples]
                pt_samples = np.unique(pt_samples, axis=0)
                while len(pt_samples) < 10:
                    # id_one = self.inliers[random.sample(range(0, len(self.inliers)), 5)].tolist()
                    id_one = self.bsample.tolist()
                    del id_one[random.randint(0, len(id_one) - 1)]
                    del id_one[random.randint(0, len(id_one) - 1)]
                    id_two = np.array(random.sample(range(0, n_points), 2)).tolist()
                    id_one.insert(random.randint(0, len(id_one)), id_two[0])
                    id_one.insert(random.randint(0, len(id_one)), id_two[1])
                    id_samples = np.array(id_one)
                    pt_samples = pts[id_samples]
                    pt_samples = np.unique(pt_samples, axis=0)
            ptc = pt_samples.copy()
            ptc[0] = ptc[-1]
            ptc = np.roll(ptc, -1, axis=0)
            vv = pt_samples - ptc
            vv = vv[:-1]
            vv2 = vv.copy()
            vv2[0] = vv2[-1]
            vv2 = np.roll(vv2, -1, axis=0)
            # vectera->b
            aa = vg.angle(vv, vv2)
            aa = aa[:-1]
            # aa=np.abs(aa)
            print(np.min(aa), np.max(aa))
            if np.max(aa) > 60:
                continue
            xaxis=pt_samples[:, 0]
            yaxis=pt_samples[:, 1]
            zaxis=pt_samples[:, 2]
            curve=cubicSplineInterpolate(xaxis,yaxis,zaxis)
            tree2 = spatial.cKDTree(pts)
            mindist, minid = tree2.query(curve)
            print(max(mindist),len(curve))
            if max(mindist) > 10:
                continue
            cp_curve = curve.copy()
            cp_curve[0] = cp_curve[-1]
            cp_curve = np.roll(cp_curve, -1, axis=0)
            lenth = (curve[:, 0] - cp_curve[:, 0]) ** 2 + (curve[:, 1] - cp_curve[:, 1]) ** 2 + (
                    curve[:, 2] - cp_curve[:, 2]) ** 2
            reallenth = np.sum(np.sqrt(lenth))
            segvec = curve - cp_curve
            # segvec=segvec[0:-1:2]
            segvec = segvec[:-1]
            segvec2 = segvec.copy()
            segvec2[0] = segvec2[-1]
            segvec2 = np.roll(segvec2, -1, axis=0)
            # print(segvec[0], segvec2[0])
            angel = vg.angle(segvec, segvec2)
            angel = angel[:-1]
            if np.max(angel) > 60:
                continue
            lenth=(curve[:,0]-cp_curve[:,0])**2+(curve[:,1]-cp_curve[:,1])**2+(curve[:,2]-cp_curve[:,2])**2
            reallenth=np.sum(np.sqrt(lenth))
            tree = spatial.cKDTree(curve)
            mindist, minid = tree.query(pts)
            pt_id_inliers = []  # list of inliers ids
            pt_id_inliers = np.where(np.abs(mindist) <= thresh)[0]
            print(len(pt_id_inliers),reallenth)
            if len(pt_id_inliers)> best:
                    first = 0
                    flag=1
                    best=len(pt_id_inliers)
                    best_inliers = pt_id_inliers
                    self.inliers = best_inliers
                    self.A=curve
                    self.bsample = id_samples
                    self.supoint=pt_samples
        rootpath = "/mnt/storage/buildwin/desk_backword"
        np.save(rootpath + "/11.13/supoint.npy",self.supoint)
        return self.A,self.inliers
