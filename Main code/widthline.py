import random
import open3d as o3d
import numpy as np
import vg
from scipy import spatial

from cubic import cubicSplineInterpolate


class WLine:
    """
    Implementation for 3D Line RANSAC.

    This object finds the equation of a line in 3D space using RANSAC method.
    This method uses 2 points from 3D space and computes a line. The selected candidate will be the line with more inliers inside the radius theshold.

    ![3D line](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/line.gif "3D line")

    ---
    """

    def __init__(self):
        self.best_inliers=[]
        self.inliers = []
        self.A = []
        self.B = []

    def fit(self, pcl,pts,curve,threedcurve,neigbor,ang):
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

        def rotation_matrix_from_vectors(vec1, vec2):
            a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
            return rotation_matrix
        def corss(lp, rp, stp, edp):
            # rm = rotation_matrix_from_vectors(np.array([0, 0, 1]), E[:3])
            # print(stp,rm,E)
            # stp = np.dot(stp, rm)[:2]
            # edp = np.dot(edp, rm)[:2]
            # rp = np.dot(rp, rm)[:2]
            # lp= np.dot(lp, rm)[:2]
            stp=stp[:2]
            edp=edp[:2]
            rp=rp[:2]
            lp=lp[:2]
            v1 = lp - stp
            v2 = rp - stp
            v3 = edp - stp
            v4 = lp - rp
            v5 = stp - rp
            v6 = edp - rp
            a3 = np.cross(v5, v4)
            a4 = np.cross(v6, v4)
            a1 = np.cross(v1, v3)
            a2 = np.cross(v2, v3)

            if a1 * a2 > 0 or a3 * a4 > 0:
                return False
            return True

        n_points = pts.shape[0]
        best_inliers = []
        threepset=[]
        flag=-1
        for it in range(1,len(curve)-1):
            blp=[0,0,0]
            brp=[0,0,0]
            stp=curve[it-1]
            edp=curve[it+1]
            dist_pt = np.sqrt((curve[it][0] - pts[:, 0])**2 + ( curve[it][1] - pts[:, 1])**2 +( curve[it][2] - pts[:, 2])**2)
            v1=[stp[0]-edp[0],stp[1]-edp[1],stp[2]-edp[2]]
            #neighbor area

            def inler(neigbor):
                i=2
                cd_id_inliers = []
                print(len(cd_id_inliers),212)
                while len(cd_id_inliers)<neigbor:
                    i+=1
                    cd_id_inliers = np.where(np.abs(dist_pt) <= i)[0]
                return cd_id_inliers
            cd_id_inliers=inler(neigbor)
            #cd_id_inliers = np.where(np.abs(dist_pt) <= 15)[0]
            # print(len(cd_id_inliers))
            # wid_part = pcl.select_by_index(cd_id_inliers).paint_uniform_color([0, 1, 0])
            # o3d.visualization.draw_geometries([ wid_part ])
            if len(cd_id_inliers)<2:
                continue
            len_cdd=len(cd_id_inliers)
            itnumber=5000
            bestscore=100000
            bestptsid=[]
            while itnumber>0 or flag!=0:
                if flag==-1:
                    flag = 1
                itnumber-=1
                id_samples = random.sample(range(0, len_cdd), 2)
                p1=pts[cd_id_inliers[id_samples[0]]]
                p2=pts[cd_id_inliers[id_samples[1]]]
                lp=p1
                rp=p2
                v2=lp-rp
                if np.cross(v2[:2],v1[:2])<0:
                    lp=p2
                    rp=p1
                    v2=lp-rp

                if corss(lp,rp,stp,edp)==False:
                    # if j == itnumber - 1:
                    #     break
                    # else:
                        continue
                # print(itnumber,"pass", bestscore)
                ang=vg.angle(np.array([v1]),np.array([v2]))
                if abs(ang[0]-90)>ang:
                    # if j==itnumber-1:
                    #     break
                    # else:
                        continue
                score=np.sqrt((lp[0] - rp[0])**2 + (lp[1] - rp[1])**2 +(lp[2] - rp[2])**2)-abs(ang[0]-90)/90+\
                      np.linalg.norm(np.cross(np.array(lp-curve[it]),np.array(v2))/np.linalg.norm(np.array(v2)))
                if score<bestscore:
                    flag=0
                    blp=lp
                    brp=rp
                    bestscore=score
                    bestptsid = [cd_id_inliers[id_samples[0]],cd_id_inliers[id_samples[1]]]
            print(bestptsid,blp,brp)
            if sum(blp)!=0 and sum(brp)!=0 and flag==0:
                best_inliers.extend(bestptsid)

                # pts = np.delete(pts, bestptsid,axis=0)

                print(len(best_inliers),len(threepset),"ttttttt")
                threepset.append([curve[it],blp,brp,threedcurve[it]])
        # threepset.append([curve[-1], np.array([0, 0, 0]), np.array([0, 0, 0])])
        print(best_inliers,threepset)
        return  best_inliers,threepset