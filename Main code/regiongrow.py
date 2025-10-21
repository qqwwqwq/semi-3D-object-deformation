import math
import numpy as np
import scipy
from scipy.optimize import root


def sourond(seed,size,imgd,tre):
    cx = 957.895
    cy = 546.481
    fx = 916.797
    fy = 916.603
    def two2three(x,y,dp):
        z=int(dp[int(x)][int(y)])
        xw = (y - cx) * z / fx
        yw = (x - cy) * z / fy
        return [xw,yw,z]


    center=two2three(seed[0],seed[1],imgd)
    # Ne=[(-size,0) , (size,0) , (0,-size) , (0,size),(size,size),(size,-size),(-size,size),(-size,-size)]
    Ne=[]
    pointser=[]
    angle=scipy.linspace(0,160,9)
    for i in angle:
        # if i==0:
        #     pu=0
        # else:
        #     pu=math.tan(i*math.pi/180)
        # def f1(x):
        #     return [(x[0] - center[0]) * tre[0] + (x[1] - center[1]) * tre[1] + (x[2] - center[2]) * tre[2],
        #             (x[0]- center[0]) ** 2 + (x[1] - center[1]) ** 2 + (x[2] - center[2]) ** 2 - size * size,
        #             (x[0]-center[0])/(x[1]-center[1])-pu]
        # pointser.append(root(f1,np.array([0,0,0])).x)
        Ne.append((int(size*math.sin(i*math.pi/180)),int((size)*math.cos(i*math.pi/180))))
    print(pointser,"neeeee")
    l=[]

    # cx = 2044.08
    #
    # cy = 1550.39
    #
    # fx = 1955.83
    #
    # fy = 1955.42

    def exist(p1,p2):
        sumee=0
        xs=scipy.linspace(float(p1[0]),float(p2[0]),size*2-4)
        ys=scipy.linspace(float(p1[1]), float(p2[1]), size*2-4)
        zs=scipy.linspace(float(p1[2]),float(p2[2]),size*2-4)
        for i in range(1,len(xs)-1):
            ox=xs[i]
            oy=ys[i]
            oz=zs[i]
            yy = round(ox * fx / oz + cx)
            xx=round(oy*fy/oz+cy)
            if xx<0 or xx>1079 or yy<0 or yy>1919:
                sumee += 10
                continue
            zz=int(imgd[xx][yy])
            xw = (yy - cx) * zz / fx
            yw=(xx-cy)*zz/fy
            error=math.sqrt(pow(ox-xw,2)+pow(oy-yw,2)+pow(float(oz)-float(zz),2))
            # print(error,ox,oy,oz,xw,yw,zz)
            sumee+=round(error)
        return sumee/(len(xs)-2)
    # for n in pointser[1:]:
    #     p1=[n[0],n[1],n[2]]
    #     p2=[center[0]*2-n[0],center[1]*2-n[1],center[2]*2-n[2]]
    #     print(p1,p2)
    #     l.append(exist(p1,p2))
    for (p, q) in Ne:
        point1=two2three(seed[0]-p,seed[1]-q,imgd)
        point2=two2three(seed[0]+p,seed[1]+q,imgd)
        l.append(exist(point1,point2))
    np.asarray(l)
    return max(l)
    # def checks(a,b):
    #     o1=seed[0]
    #     o2=seed[1]
    #     cx = 957.895
    #     cy = 546.481
    #     fx = 916.797
    #     fy = 916.603
    #     x1=(b - cx) * int(imgd[a][b]) / fx
    #     y1=(a - cy) * int(imgd[a][b])/ fy
    #     x2=(o2 - cx) * int(imgd[o1][o2]) / fx
    #     y2=(o1 - cy) * int(imgd[o1][o2]) / fy
    #     out=[(x1+x2)/2,(y1+y2)/2,(float(imgd[a][b])+float(imgd[o1][o2] ))/2]
    #     yy=int(out[0]*fx/out[2]+cx)
    #     xx=int(out[1]*fy/out[2]+cy)
    #     zz=int(imgd[xx][yy])
    #     xw=(yy - cx) * zz / fx
    #     yw=(xx-cy)*zz/fy
    #     error=math.sqrt(pow(out[0]-xw,2)+pow(out[1]-yw,2)+pow(float(out[2])-float(zz),2))
    #     return error
    # for (p, q) in Ne:
    #     lero=checks(seed[0]+p,seed[1]+q)
    #     l.append(lero)
    # for i in l:
    #     if i>tre:
    #
    #         return False
    # return True