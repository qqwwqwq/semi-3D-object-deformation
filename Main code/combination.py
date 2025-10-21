import math

import random
import struct
import pyransac3d as pyrsc
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from superfacts import segment

from planegrow import growprocess
from regiongrow import sourond

import pylab
import open3d as o3d
import scipy

from mpl_toolkits.mplot3d import Axes3D

import scipy.ndimage as nd

import scipy.signal as signal

import cv2

import cv2.aruco as aruco

from matplotlib import pyplot as plt

import cv2

import vg
from MRF_new import MRF

from scipy.optimize import leastsq, fsolve

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.tri as mtri

from scipy.spatial import Delaunay

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mpl_toolkits.mplot3d import Axes3D


# from LLSnew import growprocess

# from findcorner import projecting,corners

np.set_printoptions(suppress=True)

# pd.set_option("display.float_format",lambda x:"%.5f"%x)

# with np.load('C.npz') as X:
#
#     mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
rootpath="/home/hexin/桌面"

def bgr_rgb(img):

    (r, g, b) = cv2.split(img)

    return cv2.merge([b, g, r])

def angelarea(p1,p2,p3):

    a= float(math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2)))

    b = float(math.sqrt(pow(p2[0] - p3[0], 2) + pow(p2[1] - p3[1], 2) + pow(p2[2] - p3[2], 2)))

    c = float(math.sqrt(pow(p1[0] - p3[0], 2) + pow(p1[1] - p3[1], 2) + pow(p1[2] - p3[2], 2)))

    if a+b>c and a+c>b and b+c>a:

        s = (a + b + c) / 2

        print(a, b, c, s)

        S = math.sqrt((s * (s - a) * (s - b) * (s - c)))

        print(S)

        return int(S)

    else:

        return 0

def findsand(img):

    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    parameters = aruco.DetectorParameters_create()

    outs=[]

    # 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    for i in corners:

        outs.append(np.mean(i[0], axis=0))

    return outs

# image = cv2.imread('/home/hexin/Desktop/4color/df6.png')

# dp=np.load('/home/hexin/Desktop/4color/df6.npy')

# plt.imshow(dp)

# plt.show()



fpointset=[]

pointset=[]

distanceset=[]

pointsetcol=[]

pointset2=[]

error=[]

error2=[]

error3=[]

yse=[]

xse=[]

zse=[]

xx1=[]

yy1=[]

zz1=[]

def Least_squares(x,y):

    x_ = x.mean()

    y_ = y.mean()

    m = np.zeros(1)

    n = np.zeros(1)

    k = np.zeros(1)

    p = np.zeros(1)

    for i in np.arange(len(x)):

        k = (x[i]-x_)* (y[i]-y_)

        m += k

        p = np.square( x[i]-x_ )

        n = n + p

    a = m/n

    b = y_ - a* x_

    return a,b

def randomcolor():

    colora=()

    for i in range(3):

        colora=colora+(random.randint(0,255),)

    return colora

def linear_fitting_3D_points(points):

    # 表示矩阵中的值

    Sum_X = 0.0

    Sum_Y = 0.0

    Sum_Z = 0.0

    Sum_XZ = 0.0

    Sum_YZ = 0.0

    Sum_Z2 = 0.0

    av=0



    for i in range(0, len(points)):

        xi = points[i][0]

        yi = points[i][1]

        zi = points[i][2]



        Sum_X = Sum_X + xi

        Sum_Y = Sum_Y + yi

        Sum_Z = Sum_Z + zi

        Sum_XZ = Sum_XZ + xi * zi

        Sum_YZ = Sum_YZ + yi * zi

        Sum_Z2 = Sum_Z2 + zi ** 2



    n = len(points)  # 点数

    den = n * Sum_Z2 - Sum_Z * Sum_Z  # 公式分母

    if den==0:

        return 0

    k1 = (n * Sum_XZ - Sum_X * Sum_Z) / den

    b1 = (Sum_X - k1 * Sum_Z) / n

    k2 = (n * Sum_YZ - Sum_Y * Sum_Z) / den

    b2 = (Sum_Y - k2 * Sum_Z) / n

    z0=points[0][2]-1

    x0=k1*z0+b1

    y0=k2*z0+b2

    m=np.array([k1,k2,1])

    for i in points:

        ap=np.array([i[0]-x0,i[1]-y0,i[2]-z0])

        d=abs(np.dot(ap,m))/math.sqrt(k1**2+k2**2+1)

        print((i[0]-x0)**2+(i[1]-y0)**2+(i[2]-z0)**2,d**2)

        if (i[0]-x0)**2+(i[1]-y0)**2+(i[2]-z0)**2-d**2<=0:

            av+=0

        else:

            h=math.sqrt((i[0]-x0)**2+(i[1]-y0)**2+(i[2]-z0)**2-d**2)

            print(h)

            av+=h

    return av/len(points)

def lines(face,start,nn):

    print(face)

    def f(n):

        x = float(n[0])

        y = float(n[1])

        z = float(n[2])

        return [

            (start[0] - x) ** 2 + (start[1] - y) ** 2 + (start[2] - z) ** 2 - nn ** 2,

            (x-start[0])/face[0][0]-(y-start[1])/face[0][1],

            ((y-start[1])/face[0][1]+(z-start[2]))

        ]



    result1 = fsolve(f, [1, 1, 1])

    return result1

def two2three(x,y,dp):

    z=int(dp[int(x)][int(y)])
    #
    # cx = 2044.08
    #
    # cy = 1550.39
    #
    # fx = 1955.83
    #
    # fy = 1955.42
    cx = 956.09710693

    cy = 544.57885742

    fx = 918.08453369

    fy = 918.0814209
    # cx = 637.23138428
    #
    # cy = 362.88589478
    #
    # fx = 612.05633545
    #
    # fy = 612.05432129
    # cx = 1274.96276855
    #
    # cy = 726.27178955
    #
    # fx = 1224.1126709
    #
    # fy = 1224.10864258

    xw = (y - cx) * z / fx

    yw = (x - cy) * z / fy

    return [xw,yw,z]



def distance(a,b):

    s= math.sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2)+pow(a[2]-b[2],2))

    return s

def distance2(a,b):

    s= math.sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2))

    return s

def LLs(x, y, z, size):

    a = 0

    A = np.ones((size, 3))

    for i in range(0, size):

        A[(i, 0)] = x[a]

        A[(i, 1)] = y[a]

        a = a + 1



    b = np.zeros((size, 1))

    a = 0

    for i in range(0, size):

        b[(i, 0)] = z[a]

        a = a + 1

    try:

        A_T = A.T

        A1 = np.dot(A_T, A)

        A2 = np.linalg.inv(A1)

        A3 = np.dot(A2, A_T)

        X = np.dot(A3, b)

        for i in X:

            i[0] = float(i[0])

        return X

    except:

        print("矩阵不存在逆矩阵")

        return []



def panarl(ps):
    x=[]
    y=[]
    z=[]
    for i in ps:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])


    A=LLs(x,y,z,len(z))

    if len(A)>0:

        dir = [A[0][0], A[1][0], -1]

        distan = A[2][0]

        return [dir,distan]

    else:

        return []
def planeerror(plane,point):

    n=plane[0]

    d=plane[1]
    sum=0
    e=pow(point[0]*n[0]+point[1]*n[1]+point[2]*n[2]+d,2)
    return e

def planeerrorav(plane,pointset):

    n=plane[0]

    d=plane[1]

    sum=0

    for i in pointset:

        e=pow(i[0]*n[0]+i[1]*n[1]+i[2]*n[2]+d,2)

        sum+=e

    return sum / len(pointset)

def offset(p,dp):

    cx = 2044.08

    cy = 1550.39

    fx = 1955.83

    fy = 1955.42

    yy = int(p[0] * fx / p[2] + cx)

    xx = int(p[1] * fy / p[2] + cy)

    if xx > 3071 or yy > 4095 or xx < 0 or yy < 0:

        return 9999

    zz = int(dp[xx][yy])

    xw = (yy - cx) * zz / fx

    yw = (xx - cy) * zz / fx

    error = pow(p[0] - xw, 2) + pow(p[1] - yw, 2) + pow(p[2] - zz, 2)

    return error, (p[0] - xw+ p[1] - yw)



def surface_curvature(X, Y, Z):
    (lr, lb) = X.shape
    k=1
    # First Derivatives
    Xv, Xu = np.gradient(X,k)
    Yv, Yu = np.gradient(Y,k)
    Zv, Zu = np.gradient(Z,k)

    # Second Derivatives
    Xuv, Xuu = np.gradient(Xu,k)
    Yuv, Yuu = np.gradient(Yu,k)
    Zuv, Zuu = np.gradient(Zu,k)

    Xvv, Xuv = np.gradient(Xv,k)
    Yvv, Yuv = np.gradient(Yv,k)
    Zvv, Zuv = np.gradient(Zv,k)

    # 2D to 1D conversion
    # Reshape to 1D vectors
    Xu = np.reshape(Xu, lr * lb)
    Yu = np.reshape(Yu, lr * lb)
    Zu = np.reshape(Zu, lr * lb)
    Xv = np.reshape(Xv, lr * lb)
    Yv = np.reshape(Yv, lr * lb)
    Zv = np.reshape(Zv, lr * lb)
    Xuu = np.reshape(Xuu, lr * lb)
    Yuu = np.reshape(Yuu, lr * lb)
    Zuu = np.reshape(Zuu, lr * lb)
    Xuv = np.reshape(Xuv, lr * lb)
    Yuv = np.reshape(Yuv, lr * lb)
    Zuv = np.reshape(Zuv, lr * lb)
    Xvv = np.reshape(Xvv, lr * lb)
    Yvv = np.reshape(Yvv, lr * lb)
    Zvv = np.reshape(Zvv, lr * lb)

    Xu = np.c_[Xu, Yu, Zu]
    Xv = np.c_[Xv, Yv, Zv]
    Xuu = np.c_[Xuu, Yuu, Zuu]
    Xuv = np.c_[Xuv, Yuv, Zuv]
    Xvv = np.c_[Xvv, Yvv, Zvv]

    # First fundamental Coeffecients of the surface (E,F,G)
    E = np.einsum('ij,ij->i', Xu, Xu)
    F = np.einsum('ij,ij->i', Xu, Xv)
    G = np.einsum('ij,ij->i', Xv, Xv)

    m = np.cross(Xu, Xv, axisa=1, axisb=1)
    p = np.sqrt(np.einsum('ij,ij->i', m, m))
    n = m / np.c_[p, p, p]  # n is the normal

    # Second fundamental Coeffecients of the surface (L,M,N), (e,f,g)
    L = np.einsum('ij,ij->i', Xuu, n)  # e
    M = np.einsum('ij,ij->i', Xuv, n)  # f
    N = np.einsum('ij,ij->i', Xvv, n)  # g

    # Gaussian Curvature
    # Alternative formula for gaussian curvature in wiki
    # K = det(second fundamental) / det(first fundamental)
    K = (L * N - M ** 2) / (E * G - F ** 2)
    K = np.reshape(K, lr * lb)
    # wiki trace of (second fundamental)(first fundamental inverse)
    for i in range(len(K)):
        if np.isnan(K[i]):
            K[i]=0
    # Mean Curvature
    H = ((E * N + G * L - 2 * F * M) / ((E * G - F ** 2))) / 2
    H = np.reshape(H, lr * lb)
    for i in range(len(H)):
        if np.isnan(H[i]):
            H[i]=0

    # Principle Curvatures
    Pmax = H + np.sqrt(H ** 2 - K)
    Pmin = H - np.sqrt(H ** 2 - K)

    for i in range(len(Pmax)):
        if np.isnan(Pmin[i]):
            print(Pmax[i],np.sqrt(H ** 2 - K)[i],H[i])

    # Curvedness
    # 3D Shape Modeling for Cell Nuclear Morphological Analysis and Classification
    CV = np.sqrt((Pmin ** 2 + Pmax ** 2) / 2)

    # Shape Index
    # 3D Shape Modeling for Cell Nuclear Morphological Analysis and Classification
    SI = np.zeros(Pmax.shape)
    idx = (Pmax - Pmin) != 0
    SI[idx] = (2 / np.pi) * np.arctan((Pmin[idx] + Pmax[idx]) / (Pmax[idx] - Pmin[idx]))

    # Reshape
    K = np.reshape(K, (lr, lb))
    H = np.reshape(H, (lr, lb))
    Pmax = np.reshape(Pmax, (lr, lb))
    Pmin = np.reshape(Pmin, (lr, lb))
    CV = np.reshape(CV, (lr, lb))
    SI = np.reshape(SI, (lr, lb))

    return K, H, [Pmax, Pmin], CV, SI

def point2area_distance(face, point4):

    """

    :param point1:数据框的行切片，三维

    :param point2:

    :param point3:

    :param point4:

    :return:点到面的距离

    """

    Ax=face[0][0]

    By=face[0][1]

    Cz=-1

    D = face[1]

    mod_d = Ax * point4[0] + By * point4[1] + Cz * point4[2] + D

    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))

    d = abs(mod_d) / mod_area

    return d



################################### ------8-directions

def twod(a,b):

    s= math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))

    return s





def checkexsit(p1,p2,dp,t):

    intval = int(distance(p1,p2))

    print(intval)

    if intval<5:
        t=intval/2
    else:
        t=5
    tar=0

    for i in range(0,intval):

        p = [p1[0] + (p2[0] - p1[0]) * i / intval, p1[1] + (p2[1] - p1[1]) * i / intval,p1[2] + (p2[2] - p1[2]) * i / intval]

        if p[2]!=0 and offset(p, dp) > t:

            tar += 1

        if tar >= t:

            return False

    return True

def mgod(depth):

    #y-i,x-j



    sb=depth.copy()

    for i in range(1, len(depth) - 1):

        for j in range(1, len(depth[0]) - 1):

            print(i)

            sb[i][j]=float(math.sqrt((float(depth[i+1][j])-float(depth[i-1][j]))**2+(float(depth[i][j+1])-float(depth[i][j-1]))**2))*20000



    return sb

def dgod(corrdinate,dp):

    #y-i,x-j

    t=1

    i=corrdinate[0]

    j=corrdinate[1]

    while float(dp[i][j+t])-float(dp[i][j-t])==0:

        t+=1

    s=math.atan((float(dp[i+t][j])-float(dp[i-t][j]))/(float(dp[i][j+t])-float(dp[i][j-t])))/np.pi*180

    return s

def ogod(corrdinate,dp):

    t=1

    p1 = two2three(corrdinate[0]+t, corrdinate[1], dp)

    p2 = two2three(corrdinate[0]-t, corrdinate[1], dp)

    p3 = two2three(corrdinate[0] , corrdinate[1]+t, dp)

    p4 = two2three(corrdinate[0], corrdinate[1]-t, dp)

    px=[(p1[0]+p2[0])/2,(p1[1]+p2[1])/2,(p1[2]+p2[2])/2]

    py=[(p3[0]+p4[0])/2,(p3[1]+p4[1])/2,(p3[2]+p4[2])/2]

    rs=math.sqrt(offset(px,dp)+offset(py,dp))

    print(rs,offset(px,dp),offset(py,dp))

    return rs
dppp = cv2.imread(rootpath+'/11.13/6/cool2.png')
lab_matrix=np.ones((len(dppp),len(dppp[0])))
labling_m=np.zeros((len(dppp),len(dppp[0])))

def curvelocal(p,size,dp,size2,uu,dd,ll,rr):
    x=[]
    y=[]
    z=[]
    ps=[]
    if p[0]-size2<uu or p[0]+size2>dd or p[1]-size2<ll or p[1]+size2>rr:
        return [0,0]
    for i in range(p[0]-size,p[0]+size+1):
        for j in range(p[1]-size,p[1]+size+1):
            pw = two2three(i, j, dp)
            x.append(pw[0])
            y.append(pw[1])
            z.append(pw[2])
    xx = np.array(x,dtype=np.double)
    yy = np.array(y,dtype=np.double)
    zz=np.array(z,dtype=np.double)
    # pl=panarl(ps)
    # perror=planeerror(pl,two2three(p[0], p[1], dp))
    # perrorav=planeerrorav(pl,ps)
    xx.resize((2*size+1,2*size+1))
    yy.resize((2*size+1,2*size+1))
    zz.resize((2*size+1,2*size+1))
    print(xx.shape,1111111111111111111)
    # zz=zz.T
    curve = surface_curvature(xx, yy, zz)
    # if sum(curve[0]) == 0 and sum(curve[1]) == 0:
    #     lab_matrix[p[0] - size:p[0] + size + 1, p[1] - size:p[1] + size + 1] = np.zeros((size * 2 + 1, size * 2 + 1))
    curve[0].resize((2*size+1,2*size+1))
    curve[1].resize((2 * size + 1, 2 * size + 1))
    # labling_m[p[0],p[1]]=labletype(curve[0][size][size],curve[1][size][size])
    # print(curve[0])
    # print(curve[0][size][size])
    kc1=curve[0][size][size]
    kc2=curve[1][size][size]
    km1=np.mean(curve[0])
    km2=np.mean(curve[1])
    print(math.sqrt((kc1- km1) ** 2 + (kc2 - km2) ** 2),111111)
    return [kc1,kc2]

def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tobytes() ,rgb_points[i,1].tobytes() ,
                                        rgb_points[i,2].tobytes() )))
    fid.close()

def output(image,dp,frame,dpp,points):
#deform 1
    f=cv2.imread('/home/hexin/桌面/cool2.png' )
    plt.show()
    sa = image.copy()
    pointss=[]
    k1s=np.ones((len(dp),len(dp[0])))
    k2s=np.ones((len(dp),len(dp[0])))
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            pointss.append(two2three(i,j,dp))

    sdd = dpp.copy()
    si = image.copy()
    sid = dp.copy()
    four=findsand(si)
    g1 = four[0]
    b1 = four[1]
    y1 = four[2]
    r1 = four[3]
    four2=findsand(sa)
    g2 = four2[0]
    b2 = four2[1]
    y2= four2[2]
    r2 = four2[3]
    if len(g1) == 0 or len(b1) == 0 or len(y1) == 0 or len(r1) == 0:
        print("reference point missed-tar")
        return
    if len(g2) == 0 or len(b2) == 0 or len(y2) == 0 or len(r2) == 0:
        print("reference point missed-ini")
        return
    gap=90
    ll = int(min(g1[0], b1[0], y1[0], r1[0]))+gap
    rr = int(max(g1[0], b1[0], y1[0], r1[0]))-gap
    uu = int(min(g1[1], b1[1], y1[1], r1[1]))+gap
    dd = int(max(g1[1], b1[1], y1[1], r1[1]))-gap
    cv2.rectangle(image,(ll,uu),(rr,dd),(0,0,255),3,8)
    cv2.imwrite(rootpath+'/ccsss.png', image)
    # plt.imshow(image)
    # plt.show()
    XYZ=[]
    points=np.array(pointss)
    px = np.array(points[:, 0])
    py = np.array(points[:, 1])
    pz = np.array(points[:, 2])
    px.resize((len(image), len(image[0])))
    py.resize((len(image), len(image[0])))
    pz.resize((len(image), len(image[0])))
    px=px.astype(np.float)
    py=py.astype(np.float)
    pz=pz.astype(np.float)
    sxx=px[uu:dd,ll:rr]
    syy=py[uu:dd,ll:rr]
    szz=pz[uu:dd,ll:rr]
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    # ax=Axes3D(fig)
    manager=plt.get_current_fig_manager()
    manager.window.showMaximized()
    # ax.plot_surface(sxx,syy,szz)
    plt.axis("off")
    plt.subplot(121)
    plt.axis("off")
    plt.imshow(bgr_rgb(si))
    plt.subplot(122)
    plt.axis("off")

    plt.imshow(bgr_rgb(f))
    plt.show()
    plt.axis("off")
    manager=plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.imshow(bgr_rgb(image))
    plt.show()
    _,_,curve,CV,SI = surface_curvature(sxx, syy, szz)


    # curve[0].resize((dd-uu,rr-ll))
    # curve[1].resize((dd-uu,rr-ll))
    # print(curve[0].shape)
    k1s[uu:dd,ll:rr]=curve[0]
    k2s[uu:dd,ll:rr]=curve[1]
    sxx=sxx.flatten()
    syy=syy.flatten()
    szz=szz.flatten()
    d=np.vstack((sxx,syy))
    o=np.vstack((d,szz)).T
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(o)
    # pcd=pcd.voxel_down_sample(voxel_size=0.002)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(o)
    # pcd.paint_uniform_color([0, 1, 0])
    # pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(100)
    # radii = [0.5, 1, 2, 4]
    # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([rec_mesh])
    # rec_mesh.compute_vertex_normals()
    # pcd = rec_mesh.sample_points_poisson_disk(200000)
    # o3d.visualization.draw_geometries([pcd])
    plano1 = pyrsc.Plane()
    write_pointcloud(rootpath+'/11.13/ts.ply',np.array(pcd.points))
    E,B=plano1.fit(np.asarray(pcd.points), thresh=3, maxIteration=1000)
    p40 = pcd.select_by_index(B).paint_uniform_color([0, 1, 0])
    p4 = pcd.select_by_index(B, invert=True).paint_uniform_color([0, 0, 1])
    # o3d.visualization.draw_geometries([p40,p4])
    # o3d.visualization.draw_geometries([pcd])
    pcd.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100))
    # o3d.visualization.draw_geometries([pcd],point_show_normal=True)
    normalset=np.asarray(pcd.normals)
    normalset.resize((dd-uu,rr-ll,3))
    pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100))
    # o3d.visualization.draw_geometries([pcd2],point_show_normal=True)
    # normalset_S=np.asarray(pcd2.normals)
    # normalset_S.resize((dd-uu,rr-ll,3))
    normalset_S=np.zeros((dd-uu,rr-ll,4))
    redio=np.zeros((dd-uu,rr-ll,4))
    redio2=np.zeros((dd-uu,rr-ll,4))
    redis=np.zeros((dd-uu,rr-ll,1))
    print(normalset[0])
    print(normalset_S[0])
    # supers=image.copy()
    # segment(uu,dd,ll,rr,rootpath+'/11.13/w1.ply',supers)
    # cv2.imwrite(rootpath+'/11.13/sp1.png', supers)
    # segment(uu,dd,ll,rr,rootpath+'/11.13/w2.ply',supers)
    # cv2.imwrite(rootpath+'/11.13/sp2.png', supers)
    # segment(uu,dd,ll,rr,rootpath+'/11.13/w3.ply',supers)
    # cv2.imwrite(rootpath+'/11.13/sp3.png', supers)
    # segment(uu,dd,ll,rr,rootpath+'/11.13/w4.ply',supers)
    # cv2.imwrite(rootpath+'/11.13/sp4.png', supers)
    # return
    # def angle(v1, v2):
    #     angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    #     print(angle,"--------------")
    #     return abs(angle)

    #         if dp[i][j]!=0 :
    #             dir,seed,dpp=growprocess([[i,j]],dp,uu,dd,rr,ll)
    #             dp=dpp
    #             if dir.all()==np.array([0,0,0]).all():
    #                 continue
    #             for t in seed:
    #                 if t[0] <dd and t[1]<rr:
    #                     print("yes")
    #                     normalset_S[t[0] - uu][t[1] - ll]=dir
    #         if dp[uu:dd,ll:rr].max()==0:
    #             break
    # plt.imshow(dp)
    # plt.show()
    def adddem(pcloud):
        # max=pcloud.get_max_bound()
        # min=pcloud.get_min_bound()
        # df=max-min
        # if df[-1]==0 or df[0]==0 or df[1]==0:
        c = np.array(pcloud.points)
        for i in c:
            i += np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(c)
        p3_load = np.concatenate((np.asarray(pcloud.points), np.asarray(pc.points)), axis=0)
        new = o3d.geometry.PointCloud()
        new.points = o3d.utility.Vector3dVector(p3_load)
        return new
        # return pcloud
    ts=0
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd.paint_uniform_color([0, 1, 0])
    for i in range(uu, dd):
        for j in range(ll, rr):
            if  k1s[i][j] == 0 and k2s[i][j] == 0:
                continue
            if i==uu:
                id=j-ll
            else:
                id=(i-uu)*(rr-ll)+j-ll
            # [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[id],100)
            [k, idx, _] =pcd_tree.search_radius_vector_3d(pcd.points[id],5)
            currp=np.array([px[i][j],py[i][j],pz[i][j]])
            planos = pyrsc.Plane()
            pcdnn= pcd.select_by_index(idx)
            pcdnn.paint_uniform_color([0, 0, 1])
            E1, B1 = planos.fit(np.asarray(pcdnn.points), thresh=0.1, maxIteration=50)
            normalset_S[i - uu][j - ll] = E1

            ts+=1
            print(ts)
            p40 = pcdnn.select_by_index(B1).paint_uniform_color([1, 0, 0])
            p4 = pcdnn.select_by_index(B1, invert=True).paint_uniform_color([0, 0, 1])
            # if i >uu+40and j>ll+140:
            #     o3d.visualization.draw_geometries([p4,p40])
            #     o3d.visualization.draw_geometries([pcd, pcdnn])
            if len(np.array(p4.points))<5:
                redio[i - uu][j - ll] = E1
                redis[i - uu][j - ll] =0

            else:
                planos = pyrsc.Plane()
                E2, B2 = planos.fit(np.asarray(p4.points), thresh=0.2, maxIteration=30)
                if len(E2)==0:
                    redio[i - uu][j - ll] = E1
                    redis[i - uu][j - ll] = 0
                    continue
                redio[i - uu][j - ll] = E2
                # p50=p4.select_by_index(B2).paint_uniform_color([0,1, 0])
                p5=p4.select_by_index(B2, invert=True).paint_uniform_color([0, 0, 1])
                # bord5=adddem(p50).get_oriented_bounding_box()
                # bord4 = adddem(p40).get_oriented_bounding_box()
                # bord5.color=(0,0.4,0.2)
                # bord4.color = (0.5, 0.1, 0)
                # if i > uu + 40 and j > ll + 140:
                #     o3d.visualization.draw_geometries([p5, p50])
                #     o3d.visualization.draw_geometries([p50,p40,bord5,bord4])
                if len(np.array(p5.points)) > 10:
                    planos = pyrsc.Plane()
                    E3, B3 = planos.fit(np.asarray(p5.points), thresh=0.2, maxIteration=30)
                    if len(E3) == 0:
                        redio2[i - uu][j - ll] = E2
                        continue

                    redio2[i - uu][j - ll] = E3
            #         p60 = p5.select_by_index(B3).paint_uniform_color([0, 1, 1])
            #         p6 = p5.select_by_index(B3, invert=True).paint_uniform_color([0, 0, 1])
            #         bord6 = adddem(p60).get_oriented_bounding_box()
            #         bord6.color = (0.5, 0.4, 0.7)
            #         if i > uu + 35 and j > ll + 230:
            #             o3d.visualization.draw_geometries([p6, p60])
            #             o3d.visualization.draw_geometries([p50, p40,p60,bord5,bord4,bord6])
            #     d1=(
            #     E1[0] * currp[0] + E1[1] * currp[1] + E1[2] *currp[2] + E1[3]
            # ) / np.sqrt(E1[0] ** 2 + E1[1] ** 2 + E1[2] ** 2)
            #     d2=(
            #     E2[0] * currp[0] + E2[1] * currp[1] + E2[2] *currp[2] + E2[3]
            # ) / np.sqrt(E2[0] ** 2 + E2[1] ** 2 + E2[2] ** 2)
            #     print(d1,d2)
            #     redis[i - uu][j - ll] = max(abs(d1),abs(d2))
            #



    # #
    aset=np.zeros((dd-uu,rr-ll))
    # for i in range(uu, dd):
    #     for j in range(ll, rr):
    #         if  k1s[i][j] == 0 and k2s[i][j] == 0:
    #             continue
    #         # XYZ.append(two2three(i,j,dp))
    #         Ein=normalset_S[i - uu][j - ll]
    #         print(Ein)
    #
    #
    #         # angel = vg.angle(np.array(normalset_S[i - uu][j - ll]), np.array(normalset[i - uu][j - ll]))
    #         angel=abs(Ein[0]*px[i][j]+Ein[1]*py[i][j]+Ein[2]*pz[i][j]+Ein[3])
    #         print(angel,"error")
    #         aset[i - uu][j - ll] = angel
    #         # aset[i - uu][j - ll]=min(angel,180-angel)
    #         #or angle(normalset[i-uu][j-ll],[0,0,1])<30/180*math.pi
    #         # if np.array(normalset_S[i - uu][j - ll]).all() != np.array(normalset[i - uu][j - ll]).all():
    # meana=aset.mean()

    for i in range(uu, dd):
        for j in range(ll, rr):
            if  k1s[i][j] == 0 and k2s[i][j] == 0:
                continue
            size2 = 2
            # angel=vg.angle(np.array(normalset[i-uu][j-ll]), np.array(E[:3]))
            # angel=min(angel, 180 - angel)
            # ang=redio[i - uu][j - ll]
            ang1= vg.angle(np.array(normalset_S[i - uu][j - ll][:3]), np.array(redio[i - uu][j - ll][:3]))
            ang2=vg.angle(np.array(normalset_S[i - uu][j - ll][:3]), np.array(redio2[i - uu][j - ll][:3]))
            ang3=vg.angle( np.array(redio[i - uu][j - ll][:3]), np.array(redio2[i - uu][j - ll][:3]))
            ang1=min(ang1, 180 - ang1)
            ang2= min(ang1, 180 - ang2)
            ang3 = min(ang1, 180 - ang3)

            # if ang!=0 and (180-ang)!=0:
            #         print(meana)
            print(ang1,redis[i - uu][j - ll])
            if i - size2 < uu or i + size2 > dd or j - size2 < ll or j + size2 > rr \
                            or max(ang1,ang2,ang3)<25:
                        k1s[i][j] = 0
                        k2s[i][j] = 0
                        print("same")
                        continue
####################################################
            # oll=sourond([i,j],size2,dp,normalset[i-uu][j-ll])
            # print(oll)
            # if oll<1.2:
            #     k1s[i][j]=0
            #     k2s[i][ j] = 0
            #     print("same")
            # kk=curvelocal([i, j],1, dp,6,uu,dd,ll,rr)
            #
            # k1s[i][j]=kk[0]
            # k2s[i][j]=kk[1]
        # print(k1s[uu:dd,ll:rr])
    # # # # #10
    # # # #
    # np.save(rootpath+'/k1s-6.npy', k1s)
    # np.save(rootpath+'/k2s-6.npy', k2s)
    # k1s=np.load(rootpath+"/k1s-6.npy")
    # k2s=np.load(rootpath+"/k2s-6.npy")
    lbsel=MRF(image[uu:dd,ll:rr].copy(),k1s[uu:dd,ll:rr],k2s[uu:dd,ll:rr])
    xyz1=[]
    xyz2=[]
    xyz3=[]
    xyz4=[]
    # write_pointcloud(rootpath+'/11.13/ww.ply',np.array(XYZ))
    tt=image.copy()
    tt1 = image.copy()
    tt2 = image.copy()
    tt3 = image.copy()
    tt4 = image.copy()
    q1=randomcolor()
    q2=randomcolor()
    q3=randomcolor()
    q4=randomcolor()
    q1=(123,22,122)
    q2=(23,220,122)
    q3=(253,245,222)
    q4=(123,22,22)
    for i in range(len(lbsel)):
        for j in range(len(lbsel[0])):
            if lbsel[i][j]==0:
                xyz1.append([px[i+uu][j+ll],py[i+uu][j+ll],pz[i+uu][j+ll]])
                tt[i + uu][j + ll] = np.array([q1[0], q1[1], q1[2]])
                tt1[i + uu][j + ll] = np.array([q1[0], q1[1], q1[2]])
            elif lbsel[i][j]==1:
                xyz2.append([px[i+uu][j+ll],py[i+uu][j+ll],pz[i+uu][j+ll]])
                tt[i + uu][j + ll] = np.array([q2[0], q2[1], q2[2]])
                tt2[i + uu][j + ll] = np.array([q2[0], q2[1], q2[2]])
            elif lbsel[i][j]==2:
                xyz3.append([px[i+uu][j+ll],py[i+uu][j+ll],pz[i+uu][j+ll]])
                tt[i + uu][j + ll] = np.array([q3[0], q3[1], q3[2]])
                tt3[i + uu][j + ll] = np.array([q3[0], q3[1], q3[2]])
            elif lbsel[i][j]==3:
                xyz4.append([px[i+uu][j+ll],py[i+uu][j+ll],pz[i+uu][j+ll]])
                tt[i + uu][j + ll] = np.array([q4[0], q4[1], q4[2]])
                tt4[i + uu][j + ll] = np.array([q4[0], q4[1], q4[2]])
    nm=49
    cv2.imwrite(rootpath+'/11.13/tt.png', tt)
    cv2.imwrite(rootpath+'/11.13/t1.png', tt1)
    cv2.imwrite(rootpath+'/11.13/t2.png', tt2)
    cv2.imwrite(rootpath+'/11.13/t3.png', tt3)
    cv2.imwrite(rootpath+'/11.13/t4.png', tt4)
    write_pointcloud(rootpath+'/11.13/'+str(nm)+'w1.ply',np.array(xyz1))
    write_pointcloud(rootpath+'/11.13/'+str(nm)+'w2.ply',np.array(xyz2))
    # write_pointcloud(rootpath+'/11.13/w3.ply',np.array(xyz3))
    # write_pointcloud(rootpath+'/11.13/w4.ply',np.array(xyz4))
    # tt2=image.copy()

    # cv2.imwrite('/home/hexin/ev2/G1.png', tt)


    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.axis("off")
    plt.imshow(bgr_rgb(tt))

    plt.show()



    for i in pointset2:

        xx1.append(i[0])

        yy1.append(i[1])

        zz1.append(i[2])

    print(pointset2, "...............")




    # plt.subplot(221)
    #
    #
    # image = bgr_rgb(image / 255)
    #
    # plt.subplot(222)
    #
    # plt.imshow(image)
    #
    # plt.subplot(223)
    #
    # plt.imshow(dp)
    #
    # plt.subplot(224)
    #
    # plt.imshow(bgr_rgb(frame))
import random
import struct

import open3d as o3d
import numpy as np
from scipy import spatial

from width import width
from widthline import WLine
from ransacspline import ransacspline
from supshape import Supshape
from newline import nLine
import matplotlib
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
rootpath = "/home/hexin/桌面"


def write_pointcloud(filename, xyz_points, rgb_points=None):
    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3, 'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert xyz_points.shape == rgb_points.shape, 'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename, 'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
                                        rgb_points[i, 0].tobytes(), rgb_points[i, 1].tobytes(),
                                        rgb_points[i, 2].tobytes())))
    fid.close()


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      )
    return inlier_cloud, outlier_cloud


print("Statistical oulier removal")


def randomcolor():
    colora = []
    for i in range(3):
        colora.append(round(random.uniform(0, 1), 3))
    return colora


def findsegline(A, B, xmax, xmin, ymax, ymin):
    if abs(A[0]) > abs(A[1]):
        t1 = (xmax - B[0]) / A[0]
        t2 = (xmin - B[0]) / A[0]
        ymax = t1 * A[1] + B[1]
        zmax = t1 * A[2] + B[2]
        ymin = t2 * A[1] + B[1]
        zmin = t2 * A[2] + B[2]
        return [[xmax, ymax, zmax], [xmin, ymin, zmin]]
    else:
        t1 = (ymax - B[1]) / A[1]
        t2 = (ymin - B[1]) / A[1]
        xmax = t1 * A[0] + B[0]
        zmax = t1 * A[2] + B[2]
        xmin = t2 * A[0] + B[0]
        zmin = t2 * A[2] + B[2]
        return [[xmax, ymax, zmax], [xmin, ymin, zmin]]


def remove(width, curve, pts, thresh,T):
    t = (T[0] * pts[:, 0] + T[1] * pts[:, 1] + T[2] * pts[:, 2] + T[3]
         ) / np.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2)
    print(pts[0], [t * T[0], t * T[1], t * T[2]][0])
    p_t = pts - np.asarray([t * T[0], t * T[1], t * T[2]]).T
    pts = p_t
    print(pts.shape)
    print(curve.shape)
    print(width.shape)
    tree = spatial.cKDTree(curve)
    mindist, minid = tree.query(pts)
    print(mindist.shape,minid.shape)
    print(minid)
    pt_id_inliers=[]
    for i in range(len(mindist)):
        print( mindist[i],pow(np.sum((width[minid[i]][0] - width[minid[i]][1]) ** 2), 0.5) / 2)
        if mindist[i]< min(pow(np.sum((width[minid[i]][0] - width[minid[i]][1]) ** 2), 0.5)/2,thresh) :
            pt_id_inliers.append(i)
    # pt_id_inliers = \
    # np.where(np.abs(mindist) <= pow(np.sum((width[minid][0] - width[minid][1]) ** 2), 0.5) / 2)[0]
    print(len(pt_id_inliers))
    return pt_id_inliers
def extra(pts, Pe):
    # tre=        ( Pe[0] * 0 + Pe[1] *0 + Pe[2] * 0 + Pe[3]
    #         ) / np.sqrt(Pe[0] ** 2 + Pe[1] ** 2 + Pe[2] ** 2)
    # print(tre,221)
    dist_pt = (Pe[0] * pts[:, 0] + Pe[1] * pts[:, 1] + Pe[2] * pts[:, 2] + Pe[3]
               ) / np.sqrt(Pe[0] ** 2 + Pe[1] ** 2 + Pe[2] ** 2)
    print(dist_pt)
    pt_id_inliers = np.where(dist_pt < 0)[0]
    print(pt_id_inliers)
    return pt_id_inliers


def trangle(pcd_p):
    pcd_p.estimate_normals()
    pcd_p.orient_normals_consistent_tangent_plane(100)
    radii = [0.5]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_p, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([rec_mesh])
    rec_mesh.compute_vertex_normals()
    pcd_p = rec_mesh.sample_points_poisson_disk(1000)
    o3d.visualization.draw_geometries([pcd_p])
    return pcd_p


def adddem(pcloud):
    # max=pcloud.get_max_bound()
    # min=pcloud.get_min_bound()
    # df=max-min
    # if df[-1]==0 or df[0]==0 or df[1]==0:
    c = np.array(pcloud.points)
    for i in c:
        i += np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(c)
    p3_load = np.concatenate((np.asarray(pcloud.points), np.asarray(pc.points)), axis=0)
    new = o3d.geometry.PointCloud()
    new.points = o3d.utility.Vector3dVector(p3_load)
    return new


def findone(pcd__, widout,win):
    # o3d.visualization.draw_geometries([pcd__])
    # o3d.visualization.draw_geometries([widout])
    # 2d curve
    plano1 = nLine()
    A, best_inliers, pt = plano1.fit(np.asarray(pcd__.points), E, thresh=4, maxIteration=100000, tre2=1.5)
    AA = o3d.geometry.PointCloud()
    AA.points = o3d.utility.Vector3dVector(A)
    ptt = o3d.geometry.PointCloud()
    ptt.points = o3d.utility.Vector3dVector(pt)
    ptt.paint_uniform_color([1, 0.7, 0.1])
    AA.paint_uniform_color(randomcolor())
    plane2 = pcd__.select_by_index(best_inliers).paint_uniform_color(randomcolor())
    p3 = ptt.select_by_index(best_inliers).paint_uniform_color(randomcolor())
    ts = adddem(widout).get_oriented_bounding_box()
    ts.color = (0, 0.4, 0.2)
    win.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([win, ts])
    # o3d.visualization.draw_geometries([widout, ts])
    # o3d.visualization.draw_geometries([pcd__, ts])
    # o3d.visualization.draw_geometries([ptt, ts])
    # not_plane2 = pcd__.select_by_index(best_inliers, invert=True).paint_uniform_color(randomcolor())
    o3d.visualization.draw_geometries([ptt, AA, p3, ts])
    # 3d curve
    ransc = ransacspline()
    depline, depid = ransc.fit(pts=np.asarray(plane2.points), thresh=2, maxIteration=2000)
    p = plane2.select_by_index(depid).paint_uniform_color(randomcolor())
    AA = o3d.geometry.PointCloud()
    AA.points = o3d.utility.Vector3dVector(depline)
    c=randomcolor()
    AA.paint_uniform_color(c)
    # o3d.visualization.draw_geometries([p, ts])
    o3d.visualization.draw_geometries([p, AA, ts])

    # project
    threedcurve = np.asarray(AA.points)
    t = (E[0] * threedcurve[:, 0] + E[1] * threedcurve[:, 1] + E[2] * threedcurve[:, 2] + E[3]
         ) / np.sqrt(E[0] ** 2 + E[1] ** 2 + E[2] ** 2)
    twodcurve = threedcurve - np.asarray([t * E[0], t * E[1], t * E[2]]).T

    # width
    plano2 = WLine()
    p2d = o3d.geometry.PointCloud()
    p2d.points = o3d.utility.Vector3dVector(twodcurve)
    p2d.paint_uniform_color(c)
    o3d.visualization.draw_geometries([ts, AA])
    o3d.visualization.draw_geometries([ts, p2d])
    o3d.visualization.draw_geometries([widout, p2d])

    best, threepset = plano2.fit(widout, np.asarray(widout.points), twodcurve, threedcurve)
    print(len(best))
    wid_part = widout.select_by_index(best).paint_uniform_color(randomcolor())
    wid_not = widout.select_by_index(best, invert=True).paint_uniform_color(randomcolor())
    # o3d.visualization.draw_geometries([wid_part, wid_not])
    remian = pcd__.select_by_index(
        remove(np.asarray(threepset)[:, 1:3], np.asarray(threepset)[:, 0], np.array(pcd__.points), 8,E),
        invert=True).paint_uniform_color(
        randomcolor())
    print(np.array(pcd__.points).shape)

    # o3d.visualization.draw_geometries([ts, AA])
    o3d.visualization.draw_geometries([ts, wid_part, p2d, AA])
    o3d.visualization.draw_geometries([ pcd__,remian])
    return AA, wid_part, remian, threepset, np.asarray(AA.points)
import math
import random
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
from pDMP_functions import pDMP
from demotrajctry import dmtracject
from dmpgain import gaindmp
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
import vg
from pytransform3d.rotations import matrix_from_axis_angle
from scipy.spatial.transform import Rotation as Rot
rootpath="/home/hexin/桌面"
#定义图像和三维格式坐标轴
def get_distance_from_point_to_line(point, line_point1, line_point2):
    #对于两点坐标为同一点时,返回点与点的距离
    if line_point1.all() == line_point2.all():
        point_array = np.array(point )
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    #计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    #根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance
def rotation_matrix_from_vectors(vec1, vec2):

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
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
    rotmat = np.dot(points,R.T)

    return rotmat
def rotation(v1,v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    # 根据点积反推夹角
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(dot_product)
    cross_product = np.cross(unit_v1, unit_v2)
    if cross_product.item() > 0:
        angle = 2 * np.pi - angle
    rotate_matrix = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return rotate_matrix
# fig=plt.figure()
# ax = Axes3D(fig)
def origin(threepset):
    fig = plt.figure()
    ax = Axes3D(fig)
    color2=["green","yellow"]
    color=["purple", "black"]
    plt.xlim(-90, 40)
    plt.ylim(-60, 70)
    for j in range(len(threepset)):
        threedline = threepset[j][:, 3]
        center = threepset[j][:, 0]
        setsamples = len(center)
        rp = threepset[j][:, 1]
        lp = threepset[j][:, 2]
        ax.scatter3D(rp[:, 0], rp[:, 1], rp[:, 2], marker='^')
        ax.scatter3D(lp[:, 0], lp[:, 1], lp[:, 2],marker='.')
        # for i in range(0, len(center)):
        #     ax.plot3D(np.array([rp[i][0], lp[i][0]]), np.array([rp[i][1], lp[i][1]]), np.array([rp[i][2], lp[i][2]]),
        #               c=color2[j])
        ax.plot3D(threedline[:, 0], threedline[:, 1], threedline[:, 2], color[j])
    plt.show()
def main(threepset,E):
    fig = plt.figure()
    ax = Axes3D(fig)
    color2 = ["purple", "black","yellow"]
    plt.xlim(-90, 40)
    plt.ylim(-60, 70)
    for j in range(len(threepset)):
        threedline = threepset[j][:, 3]
        center = threepset[j][:, 0]
        setsamples = len(center)
        rp = threepset[j][:, 1]
        lp = threepset[j][:, 2]
        for i in range(1, len(rp)):
            v1 = np.array(rp[i - 1] - lp[i - 1])
            v2 = np.array(rp[i] - lp[i])
            unit_v1 = v1 / np.linalg.norm(v1)
            unit_v2 = v2 / np.linalg.norm(v2)
            # 根据点积反推夹角
            dot_product = np.dot(unit_v1, unit_v2)
            angle = np.arccos(dot_product)
        resultx = []
        resulty = []
        resultz = []
        px=[]
        py=[]
        pz=[]
        pxx=[]
        pyy=[]
        pzz=[]
        result = []
        mask = 1
        for i in range(0, setsamples - 2,2):



            print(setsamples - 2, i)
            p1 = threedline[i]
            p2 = threedline[i + 2]
            print(p1, p2)
            rp1 = rp[i]
            lp1 = lp[i]
            rp2 = rp[i + 2]
            lp2 = lp[i + 2]
            # tree = spatial.cKDTree(center)
            if mask == 1:

                ampti = math.sqrt((center[i + 1][0] - rp[i + 1][0]) ** 2 + (center[i + 1][1] - rp[i + 1][1]) ** 2 + (
                        center[i + 1][2] - rp[i + 1][2]) ** 2) * 10 / 2
                # mindist, minid = tree.query([lp2])
                # ampti = get_distance_from_point_to_line(lp1, p1, p2) * 10 / 2

            else:

                ampti = math.sqrt((center[i + 1][0] - lp[i + 1][0]) ** 2 + (center[i + 1][1] - lp[i + 1][1]) ** 2 + (
                        center[i + 1][2] - lp[i + 1][2]) ** 2) * 10 / 2




                # mindist, minid = tree.query([rp2])

                # ampti = get_distance_from_point_to_line(rp1, p1, p2) * 10 / 2
            dis = int(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) * 10)

            print("distance", dis, ampti)
            # if dis==0:
            #     continue
            a = dmtracject(200, 0.1, dis)
            a.generate()
            b = gaindmp(200, 0.1, dis, a.data[:, 1])
            t, y = b.update(ampti)
            # b.show()
            print(y)
            if mask == 1:
                mask = -1
            else:
                mask = 1
                y = y * (-1)
            tt = np.linspace(0, dis, dis * 10)
            welement = np.vstack((tt, y)).T
            welement = welement / 10
            zaxis = np.array([[0] * len(welement)]).T
            welement = np.hstack((welement, zaxis))
            if E[j][:3].all() != np.array([0, 0, -1]).all() and E[j][:3].all() != np.array([0, 0, 1]).all():
                welement = rotmat(E[j][:3], np.array(welement))
            # welement=welement+(p1[0:2]-welement[0])
            print(p1[0:2], welement, 333333333)
            rt1 = rotation_matrix_from_vectors(np.array(
                [center[i + 2][0] - center[i][0], center[i + 2][1] - center[i][1], center[i + 2][2] - center[i][2]]),
                                               np.array(
                                                   [welement[-1][0] - welement[0][0], welement[-1][1] - welement[0][1],
                                                    welement[-1][2] - welement[0][2]]))
            welement1 = np.dot(welement, rt1)
            ############rotation_1
            rt = rotation_matrix_from_vectors(np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]), np.array(
                [welement1[-1][0] - welement1[0][0], welement1[-1][1] - welement1[0][1], welement1[-1][2] - welement1[0][2]]))
            # rt=rt[0:2,0:2]
            # print(rt)
            # rt=rotation(np.array(welement[-1]-welement[0]),np.array([p2[0]-p1[0],p2[1]-p1[1]]))
            # print(rt,welement)
            after_rt = np.dot(welement1, rt)
            after_rt += (p1 - after_rt[0])
            # welement += (p1 - welement[0])
            # welement1 += (p1 - welement1[0])
            ########## rotation 2
            rt2 = rotation_matrix_from_vectors(np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]),
                                               np.array([p2[0] - p1[0], p2[1] - p1[1], 0]))
            after_rt2 = np.dot(after_rt, rt2)
            after_rt2 += (p1 - after_rt2[0])
            resultx.extend(after_rt[:, 0])
            resulty.extend(after_rt[:, 1])
            resultz.extend(after_rt[:, 2])
            # if i>1:
            #     px.extend(welement1[:, 0])
            #     py.extend(welement1[:, 1])
            #     pz.extend(welement1[:, 2])
            #     pxx.extend(welement[:, 0])
            #     pyy.extend(welement[:, 1])
            #     pzz.extend(welement[:, 2])
            # px.extend([p1[0],p2[0]])
            # py.extend([p1[1],p2[1]])
            # pz.extend([p1[2],p2[2]])
            result.extend(after_rt)
            # plt.axis("off")
            ax.plot3D(np.array(resultx), np.array(resulty), np.array(resultz), color2[j])
            # ax.plot3D(px, py, pz, 'blue')
            # ax.plot3D(pxx, pyy, pzz, 'yellow')
            # ax.scatter(px,py,pz,"black")

            ax.scatter3D(rp[:, 0], rp[:, 1], rp[:, 2], marker='^')
            ax.scatter3D(lp[:, 0], lp[:, 1], lp[:, 2], marker='.')
            ax.plot3D(threedline[:, 0], threedline[:, 1], threedline[:, 2], 'gray')

        # ax.plot3D(center[:, 0][0:i+2], np.array(center[:, 1][0:i+2]), np.array(center[:, 2][0:i+2]), 'blue')
        #     ax.plot3D(center[:, 0], center[:, 1], center[:, 2], 'gray')
        # ax.set_xlim(-120, 20)
        # ax.set_ylim(-20, 100)
        # ax.set_zlim3d(380, 420)
        # ax.scatter3D([rp1[0],lp1[0]], [rp1[1],lp1[1]], [rp1[2],lp1[2]], cmap='blue')
        #     ax.scatter3D(rp[:, 0], rp[:, 1], rp[:, 2])
        #     ax.scatter3D(lp[:, 0], lp[:, 1], lp[:, 2])
        np.save(rootpath + "/11.13/" + str(nm) + "/result"+str(j)+".npy", np.array(result))
    plt.show()
nm=49
frame=cv2.imread(rootpath+'/11.13/'+str(nm)+'/cool2.png')
dpp=np.load(rootpath+'/11.13/'+str(nm)+'/dep1111.npy')
a = cv2.imread(rootpath+'/11.13/'+str(nm)+'/cool2.png')
b=np.load(rootpath+'/11.13/'+str(nm)+'/dep1111.npy')
points=np.load(rootpath+'/11.13/'+str(nm)+'/depcloud.npy')
print(points.shape)
see=output(a,b,frame,dpp,points)

nm = 49
pcd2 = o3d.io.read_point_cloud(rootpath + "/11.13/" + str(nm) + "/w2.ply")
# pcd3 = o3d.io.read_point_cloud(rootpath+"/11.13/w3.ply")
pcd4 = o3d.io.read_point_cloud(rootpath + "/11.13/" + str(nm) + "/w1.ply")
plano1 = pyrsc.Plane()
E, B = plano1.fit(np.asarray(pcd4.points), thresh=2.5, minPoints=1000, maxIteration=1000)
print(1111111111)
p40 = pcd4.select_by_index(B).paint_uniform_color([0, 1, 0])
p4 = pcd4.select_by_index(B, invert=True).paint_uniform_color([0, 0, 1])

iner = extra(np.asarray(p4.points), E)
pout = p4.select_by_index(iner).paint_uniform_color([1, 0, 1])
pin = p4.select_by_index(iner, invert=True).paint_uniform_color([0, 1, 0])
if np.asarray(pin.points).shape[0] < np.asarray(pout.points).shape[0]:
    pout, pin = pin, pout
# o3d.visualization.draw_geometries([pin])
# cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=50,
#                                                         std_ratio=0.5)
cl2, ind2 = pcd2.remove_statistical_outlier(nb_neighbors=50,
                                            std_ratio=2)
# cl3, ind3 = pcd3.remove_statistical_outlier(nb_neighbors=50,
#                                                         std_ratio=0.5)
cl4, ind4 = pin.remove_statistical_outlier(nb_neighbors=50,
                                           std_ratio=0.01)
# a,ao=display_inlier_outlier(voxel_down_pcd, ind)
b=pcd2.select_by_index(ind2).paint_uniform_color([1, 0, 1])
d=pin.select_by_index(ind4).paint_uniform_color([0, 0, 1])
# b, bo = display_inlier_outlier(pcd2, ind2)
# # c,co=display_inlier_outlier(pcd3, ind3)
# d, do = display_inlier_outlier(pin, ind4)
#
# # a.paint_uniform_color([0, 0, 1])
# d.paint_uniform_color([1, 0.7, 0.1])
# c.paint_uniform_color([0, 1, 0])
# d.paint_uniform_color([0.5, 0, 0.5])
#
# o3d.visualization.draw_geometries([d],
#                                   )
# p3_load = np.concatenate((np.asarray(b.points),np.asarray(c.points)), axis=0)

# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
write_pointcloud(rootpath + "/11.13/" + str(nm) + "/wt.ply", np.array(b.points))
write_pointcloud(rootpath + "/11.13/" + str(nm) + "/wt2.ply", np.array(d.points))
pcd_load = o3d.io.read_point_cloud(rootpath + "/11.13/" + str(nm) + "/wt.ply")
pcd_p = o3d.io.read_point_cloud(rootpath + "/11.13/" + str(nm) + "/wt2.ply")
# pcd.paint_uniform_color([1, 0, 0])
pcd2.paint_uniform_color([0, 1, 0])
# pcd3.paint_uniform_color([0, 0, 1])
pcd_load.paint_uniform_color([0, 0, 1])
pcd_p.paint_uniform_color([1, 0.7, 0.1])
# o3d.visualization.draw_geometries([pcd_p])
# o3d.visualization.draw_geometries([pcd_load])
# pcd_p=pcd_p.voxel_down_sample(voxel_size=0.05)
# pcd_load=pcd_load.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([pcd_load, pcd_p])
points = np.asarray(pcd_load.points)
# dp=np.asarray(pcd_p.points)
# a_arg = np.argsort(dp[:, 0])
# dp = dp[a_arg].tolist()
# dp = np.asarray(dp)
# pcd_p.points=o3d.utility.Vector3dVector(dp)
# background

# pcd_pp=np.asarray(pcd_p.points)
# a_arg = np.argsort(pcd_pp[:, 0])
# pcd_pp = pcd_pp[a_arg].tolist()
# pcd_pp = np.asarray(pcd_pp)
# pcd_p.points=o3d.utility.Vector3dVector(pcd_pp)
# select width pointcloud
# width cloud---
wids = width()
widpcloud = wids.fit(np.asarray(pcd_load.points), E, thresh=2.5)
widout = pcd_load.select_by_index(widpcloud).paint_uniform_color([0, 1, 0])
dp = np.asarray(widout.points)
a_arg = np.argsort(dp[:, 0])
dp = dp[a_arg].tolist()
dp = np.asarray(dp)
t = (E[0] * dp[:, 0] + E[1] * dp[:, 1] + E[2] * dp[:, 2] + E[3]
     ) / np.sqrt(E[0] ** 2 + E[1] ** 2 + E[2] ** 2)
p_t = dp - np.asarray([t * E[0], t * E[1], t * E[2]]).T
dp = p_t
widout.points = o3d.utility.Vector3dVector(dp)
# o3d.visualization.draw_geometries([widout])
# width

# pc = o3d.geometry.PointCloud()
# pc.points = o3d.utility.Vector3dVector(A)
# pc.paint_uniform_color([0, 0, 1])
# o3d.visualization.draw_geometries([pc,widout])
# print(1)

# moved=remove(A,B,dp,thresh=5)
# remian=pcd_p.select_by_index(moved, invert=True).paint_uniform_color([0, 1, 0])
############depth
plano3 = Supshape()
#########trianglation
# pcd_p=trangle(pcd_p)
#########################
cuverset = []
widthset = []
it = 2
while it > 0:
    it -= 1
    curve, wid, remian, threepset, threedline = findone(pcd_p, widout,pcd_load)
    np.save(rootpath + "/11.13/" + str(nm) + "/" + str(it) + "_threepset_n.npy", threepset)
    np.save(rootpath + "/11.13/" + str(nm) + "/" + str(it) + "_threedline_n.npy", threedline)
    np.save(rootpath + "/11.13/" + str(nm) + "/" + str(it) + "_plane_n.npy", E)
    threepset = np.asarray(threepset)
    cuverset.append(curve)
    widthset.append(wid)
    pcd_p = remian
o3d.visualization.draw_geometries([cuverset[0], cuverset[1],widthset[0], widthset[1]])
nm=49
threepset = np.load(rootpath+"/11.13/" + str(nm) + "/0_threepset_n.npy",allow_pickle = True)
threepset2 = np.load(rootpath+"/11.13/" + str(nm) + "/1_threepset_n.npy",allow_pickle = True)
# threepset3 = np.load(rootpath+"/11.13/" + str(nm) + "/2_threepset_n.npy",allow_pickle = True)
origin([threepset,threepset2])
# threedline=np.load(rootpath+"/11.13/1_threedline.npy",allow_pickle = True)
E= np.load(rootpath+"/11.13/" + str(nm) + "/0_plane_n.npy",allow_pickle = True)
E2= np.load(rootpath+"/11.13/" + str(nm) + "/1_plane_n.npy",allow_pickle = True)
main([threepset,threepset2],[E,E2])