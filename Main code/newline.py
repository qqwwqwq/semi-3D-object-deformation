import math
import random

from matplotlib.pyplot import MultipleLocator
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from cubic import cubicSplineInterpolate
import vg
import matplotlib.colors
import matplotlib.ticker

from numpy.random import choice
class nLine:
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
    def weight(self,pts,thresh):
        n_points = pts.shape[0]
        tax=[0]*n_points
        for i in range(n_points):
            dis=np.sqrt((pts[:,0]-pts[i][0])**2+(pts[:,1]-pts[i][1])**2+(pts[:,2]-pts[i][2])**2)
            pt_id_inliers = np.where(np.abs(dis) <= thresh)[0]
            set=pts[pt_id_inliers]
            cen=np.mean(set,axis=0)
            d=np.sqrt((cen[0]-pts[i][0])**2+(cen[1]-pts[i][1])**2+(cen[2]-pts[i][2])**2)
            tax[i]=len(set)
        #     if d==0:
        #         tax[i]=1
        #     else:
        #         # while d<1:
        #         #     d*=10
        #         tax[i]=1/d
        # for t in range(len(tax)):
        #     if tax[t]==1:
        #         tax[t]=max(tax)*2
        rs = tax
            # tax[i]=1/np.mean(dis[pt_id_inliers])
        tax = np.power(tax, 8)
        # tax = np.clip(tax, 0, None)

        # m=np.mean(tax)
        # for i in range(len(tax)):
        #     if tax[i]<m:
        #         tax[i]=0
        tax=tax/np.sum(tax)
        return tax,rs

    def lerp(self,a, b, c):
        v1 = (c * a[0]) + ((1 - c) * b[0])
        v2 = (c * a[1]) + ((1 - c) * b[1])
        v3 = (c * a[2]) + ((1 - c) * b[2])
        return np.asarray([v1, v2, v3])

    def dis(self,a, b):
        k = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
        return k
    def fit(self, pts,T, thresh=4, maxIteration=1000,tre2=1,a1=30,a2=30,candi=7,w=1.5):
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
        def tostr(ids):
            s=""
            for i in ids:
               s=s+str(i)
            return s
        n_points = pts.shape[0]
        best_inliers = []
        curl=0
        best=0
        #project to plane t
        t = (T[0] * pts[:, 0] + T[1] * pts[:, 1] + T[2] * pts[:, 2] + T[3]
                  ) / np.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2)
        print(pts[0],[t*T[0],t*T[1],t*T[2]][0])
        p_t=pts-np.asarray([t*T[0],t*T[1],t*T[2]]).T
        pts=p_t
        it = maxIteration
        tax,rs=self.weight(pts,w)
        xw = pts[:, 0]
        yw = pts[:, 1]
        zw=rs
        min_v = min(zw)
        max_v = max(zw)
        color = [plt.get_cmap("seismic", len(zw))(int(float(i - min_v) / (max_v - min_v) * len(zw))) for i in zw]
        plt.set_cmap(plt.get_cmap("seismic", len(zw)))

        fig = plt.figure()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.set_major_locator(MultipleLocator(10))
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(MultipleLocator(10))
        plt.xlim(-90, 40)
        plt.ylim(-60, 70)
        im=ax.scatter3D(xw,yw,zw, c=color,s=1)
        fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda z, pos: int(z * (max_v - min_v) + min_v)),label='Number of neighbors')
        met=[]
        plt.show()
        first=1
        id_samples=np.array([])
        while it > 0:
            # print(it)
            it -= 1
            # Samples 2 random points
            # Samples 6 random points
            # id_samples = random.sample(range(0, n_points),6)
            if first<2:
                id_samples= choice(range(0, n_points),candi, p=tax)
            # while tostr(id_samples) in met:
            #     print("inininnnnnnnnnnnnnnnnnn")
            #     id_samples = choice(range(0, n_points), 6, p=tax)
                met.append(tostr(id_samples))# id_samples.sort()
                pt_samples = pts[id_samples]
                pt_samples = np.unique(pt_samples, axis=0)
                while len(pt_samples)<candi:
                    id_samples = choice(range(0, n_points), candi, p=tax)
                # id_samples.sort()
                    pt_samples = pts[id_samples]
                    pt_samples = np.unique(pt_samples, axis=0)
            else:
                # id_one=self.inliers[random.sample(range(0, len(self.inliers)),5)].tolist()
                id_one = self.bsample.tolist()
                del id_one[random.randint(0,len(id_one)-1)]
                del id_one[random.randint(0, len(id_one)-1)]
                id_two= choice(range(0, n_points),2, p=tax).tolist()
                id_one.insert(random.randint(0,len(id_one)),id_two[0])
                id_one.insert(random.randint(0, len(id_one)), id_two[1])
                id_samples=np.array(id_one)
                pt_samples = pts[id_samples]
                pt_samples = np.unique(pt_samples, axis=0)
                while len(pt_samples) < candi:
                    # id_one = self.inliers[random.sample(range(0, len(self.inliers)), 5)].tolist()
                    id_one = self.bsample.tolist()
                    del id_one[random.randint(0, len(id_one)-1)]
                    del id_one[random.randint(0, len(id_one)-1)]
                    id_two = choice(range(0, n_points), 2, p=tax).tolist()
                    id_one.insert(random.randint(0, len(id_one)), id_two[0])
                    id_one.insert(random.randint(0, len(id_one)), id_two[1])
                    id_samples = np.array(id_one)
                    pt_samples = pts[id_samples]
                    pt_samples = np.unique(pt_samples, axis=0)
            ptc = pt_samples.copy()
            ptc[0] = ptc[-1]
            ptc = np.roll(ptc, -1, axis=0)
            vv= pt_samples-ptc
            vv=vv[:-1]
            vv2=vv.copy()
            vv2[0]=vv2[-1]
            vv2 = np.roll(vv2, -1, axis=0)
            #vectera->b
            aa=vg.angle(vv,vv2)
            aa=aa[:-1]
            # print(np.min(aa), np.max(aa))
            if np.max(aa)>a1:
                continue
            #some select
            xaxis = pt_samples[:, 0]
            yaxis = pt_samples[:, 1]
            zaxis = pt_samples[:, 2]
            curve = cubicSplineInterpolate(xaxis, yaxis, zaxis)
            tree2 = spatial.cKDTree(pts)
            mindist, minid = tree2.query(curve)
            # print(max(mindist),len(curve))
            if max(mindist)>1:
                continue
            cp_curve = curve.copy()
            cp_curve[0] = cp_curve[-1]
            cp_curve = np.roll(cp_curve, -1, axis=0)
            lenth = (curve[:, 0] - cp_curve[:, 0]) ** 2 + (curve[:, 1] - cp_curve[:, 1]) ** 2 + (
                    curve[:, 2] - cp_curve[:, 2]) ** 2
            reallenth = np.sum(np.sqrt(lenth))
            segvec=curve-cp_curve
            # segvec=segvec[0:-1:2]
            segvec=segvec[:-1]
            segvec2=segvec.copy()
            segvec2[0]=segvec2[-1]
            segvec2=np.roll(segvec2, -1, axis=0)
            # print(segvec[0], segvec2[0])
            angel= vg.angle(segvec,segvec2)
            angel=angel[:-1]
            if np.max(angel)>a2:
                continue
            angelcos=np.cos(angel[:]*np.pi/180)
            tree = spatial.cKDTree(curve)
            mindist, minid = tree.query(pts)
            pt_id_inliers = []  # list of inliers ids
            pt_id_inliers = np.where(np.abs(mindist) <= thresh)[0]
            strightdis=(curve[0][0] - curve[-1][ 0]) ** 2 + (curve[0][1] - curve[-1][ 1]) ** 2 + (curve[0][2] - curve[-1][ 2]) ** 2
            print(it,strightdis,np.max(angel))
            score=len(pt_id_inliers)/len(pts)
            # -np.mean(angel) / 180
            print(score,best)
            if score >=best and reallenth>curl:
                first+=1
                curl = reallenth
                best = score
                best_inliers = pt_id_inliers
                self.inliers = best_inliers
                self.bsample = id_samples
                # ax = plt.subplot(1, 2, 1, projection='3d')
                # plt.xlim(-90, 40)
                # plt.ylim(-60, 70)
                # ax2=plt.subplot(1, 2, 2, projection='3d')
                # plt.xlim(-90, 40)
                # plt.ylim(-60, 70)
                # ax.xaxis.set_major_locator(MultipleLocator(10))
                # # 把x轴的主刻度设置为1的倍数
                # ax.yaxis.set_major_locator(MultipleLocator(10))
                # ax2.xaxis.set_major_locator(MultipleLocator(10))
                # # 把x轴的主刻度设置为1的倍数
                # ax2.yaxis.set_major_locator(MultipleLocator(10))
                #
                # ax.scatter3D(pts[id_samples][:,0], pts[id_samples][:,1], pts[id_samples][:,2], c="r",s=45)
                # ax2.scatter3D(curve[:, 0], curve[:, 1], curve[:, 2], c="b",s=1)
                # ax2.scatter3D(pts[id_samples][:, 0], pts[id_samples][:, 1], pts[id_samples][:, 2], c="r",s=45)
                #
                # plt.show()
                self.A = curve
        # result=self.A
        # res = []
        # res.append(result[0])
        # i = 0
        # j = i + 1
        # Tvalue = 2
        # while i < len(result) - 1:
        #     while self.dis(result[i], result[j]) < Tvalue and j < len(result) - 1:
        #         j = j + 1
        #     res.append(result[j])
        #     i = j
        #     j = i + 1
        # res = np.asarray(res)
        # # extend the big one
        # reset = []
        # reset.append(res[0])
        # j = 1
        # while j < len(res):
        #     lenth = self.dis(reset[-1], res[j])
        #     inter = int(lenth / Tvalue)
        #     for step in range(1, inter + 1):
        #         reset.append(self.lerp(reset[-1], res[j], step * Tvalue / lenth))
        #     j = j + 1
        # reset = np.asarray(reset)
        # self.A=reset
        tree = spatial.cKDTree(self.A)
        mindist, minid = tree.query(pts)
        pt_id_inliers = []
        pt_id_inliers = np.where(np.abs(mindist) <= tre2)[0]
        best_inliers = pt_id_inliers
        self.inliers = best_inliers

        return self.A, self.inliers,pts