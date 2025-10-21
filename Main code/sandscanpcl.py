import math

import random
import struct
import pyransac3d as pyrsc
import pandas as pd

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
    cx = 957.895

    cy = 546.481

    fx = 916.797

    fy = 916.603

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
    # print("awfshss-------------")
    # First Derivatives
    Xv, Xu = np.gradient(X)

    Yv, Yu = np.gradient(Y)

    Zv, Zu = np.gradient(Z)
    #	print(Xu)
    # Second Derivatives

    Xuv, Xuu = np.gradient(Xu)

    Yuv, Yuu = np.gradient(Yu)

    Zuv, Zuu = np.gradient(Zu)



    Xvv, Xuv = np.gradient(Xv)

    Yvv, Yuv = np.gradient(Yv)

    Zvv, Zuv = np.gradient(Zv)



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



    # % First fundamental Coeffecients of the surface (E,F,G)



    E = np.einsum('ij,ij->i', Xu, Xu)

    F = np.einsum('ij,ij->i', Xu, Xv)

    G = np.einsum('ij,ij->i', Xv, Xv)



    m = np.cross(Xu, Xv, axisa=1, axisb=1)

    p = np.sqrt(np.einsum('ij,ij->i', m, m))

    n = m / np.c_[p, p, p]

    # n is the normal

    # % Second fundamental Coeffecients of the surface (L,M,N), (e,f,g)

    L = np.einsum('ij,ij->i', Xuu, n)  # e

    M = np.einsum('ij,ij->i', Xuv, n)  # f

    N = np.einsum('ij,ij->i', Xvv, n)  # g



    # Alternative formula for gaussian curvature in wiki

    # K = det(second fundamental) / det(first fundamental)

    # % Gaussian Curvature

    K = (L * N - M ** 2) / (E * G - F ** 2)

    K = np.reshape(K, lr * lb)

    #	print(K.size)

    # wiki trace of (second fundamental)(first fundamental inverse)

    # % Mean Curvature

    H = ((E * N + G * L - 2 * F * M) / ((E * G - F ** 2))) / 2

    print(H.shape)

    H = np.reshape(H, lr * lb)

    #	print(H.size)



    # % Principle Curvatures

    Pmax = H + np.sqrt(H ** 2 - K)

    Pmin = H - np.sqrt(H ** 2 - K)

    # [Pmax, Pmin]

    Principle = [Pmax, Pmin]

    return Principle

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
    xx = np.array(x)
    yy = np.array(y)
    zz=np.array(z)
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
def compute_curvature(pcd, radius=2):

    points = np.asarray(pcd.points)

    from scipy.spatial import KDTree
    tree = KDTree(points)

    curvature = [ 0 ] * points.shape[0]

    for index, point in enumerate(points):
        indices = tree.query_ball_point(point, radius)

        # local covariance
        M = np.array([ points[i] for i in indices ]).T
        M = np.cov(M)

        # eigen decomposition
        V, E = np.linalg.eig(M)
        # h3 < h2 < h1
        h1, h2, h3 = V

        curvature[index] = h3 / (h1 + h2 + h3)

    return curvature
def output(image,dp,frame,dpp,points):
#deform 1
    sa = image.copy()
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
    gap=120
    ll = int(min(g1[0], b1[0], y1[0], r1[0]))+gap
    rr = int(max(g1[0], b1[0], y1[0], r1[0]))-gap
    uu = int(min(g1[1], b1[1], y1[1], r1[1]))+gap
    dd = int(max(g1[1], b1[1], y1[1], r1[1]))-gap
    cv2.rectangle(image,(ll,uu),(rr,dd),(0,0,255),3,8)
    cv2.imwrite(rootpath+'/ccsss.png', image)
    # plt.imshow(image)
    # plt.show()
    XYZ=[]
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
    fig=pylab.figure()
    ax=Axes3D(fig)
    ax.plot_surface(sxx,syy,szz)
    pylab.show()
    curve = surface_curvature(sxx, syy, szz)
    # print(curve[0].shape)
    curve[0].resize((dd - uu, rr - ll))
    curve[1].resize((dd - uu, rr - ll))
    k1s=curve[0].flatten()
    k2s=curve[1].flatten()
    sxx=sxx.flatten()
    syy=syy.flatten()
    szz=szz.flatten()
    d=np.vstack((sxx,syy))
    o=np.vstack((d,szz)).T
    pcd = o3d.geometry.PointCloud()
    # down_o=pcd.voxel_down_sample(voxel_size=0.002)
    o=np.unique(o,axis=0)
    pcd.points = o3d.utility.Vector3dVector(o)
    o3d.visualization.draw_geometries([pcd])
    t=compute_curvature(pcd)
    print(t)

    label=np.zeros(len(o))





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
    E,B=plano1.fit(np.asarray(pcd.points), thresh=3, maxIteration=10000)
    p40 = pcd.select_by_index(B).paint_uniform_color([0, 1, 0])
    p4 = pcd.select_by_index(B, invert=True).paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([p40,p4])
    # o3d.visualization.draw_geometries([pcd])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=200))
    o3d.visualization.draw_geometries([pcd],point_show_normal=True)
    normalset=np.asarray(pcd.normals)
    normalset_S = np.zeros(normalset.shape)
    aset = np.zeros(len(normalset))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(len(o)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 100)
        planos = pyrsc.Plane()
        pcdnn= pcd.select_by_index(idx)
        pcdnn.paint_uniform_color([0, 0, 1])
        E1, B1 = planos.fit(np.asarray(pcdnn.points), thresh=1, maxIteration=30)
        normalset_S[i] = E1[:3]


    for i in range(len(o)):
        angel = vg.angle(np.array(normalset_S[i]), np.array(normalset[i]))
        aset[i] = min(angel, 180 - angel)
    meana = aset.mean()
    for i in range(len(o)):
        ang = aset[i]
        if ang!=0 and (180-ang)!=0:
            print(meana)
            if ang<2*meana or (180-ang)<2*meana:
                k1s[i] = 0
                k2s[i] = 0
                print("same")
                continue

    for i in range(len(o)):
        k1=k1s[i]
        k2=k2s[i]
        if t[i]<0.5:
            label[i]=1
        else:
            label[i]=2
    id1=[]
    id2=[]
    for i in range(len(o)):
        if label[i]==1:
            id1.append(i)
        if label[i] == 2:
            id2.append(i)
    print(len(id2),len(id1))
    pcd2= pcd.select_by_index(id1).paint_uniform_color([0, 0, 1])
    pcd3=pcd.select_by_index(id2).paint_uniform_color([1, 0, 1])
    o3d.visualization.draw_geometries([pcd2,pcd3])

    # for i in range(uu, dd):
    #     for j in range(ll, rr):
    #         if i==uu:
    #             id=j-ll
    #         else:
    #             id=(i-uu)*(rr-ll)+j-ll
    #         [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[id],200)
    #         # [k, idx, _] =pcd_tree.search_radius_vector_3d(pcd.points[id], 2)
    #         planos = pyrsc.Plane()
    #         pcdnn= pcd.select_by_index(idx)
    #
    #         pcdnn.paint_uniform_color([0, 0, 1])
    #         E1, B1 = planos.fit(np.asarray(pcdnn.points), thresh=1, maxIteration=30)
    #         normalset_S[i - uu][j - ll] = E1[:3]
    #         ts+=1
    #         print(ts)
    #         # p40 = pcdnn.select_by_index(B1).paint_uniform_color([1, 0, 0])
    #         # p4 = pcdnn.select_by_index(B1, invert=True).paint_uniform_color([0, 0, 1])
    #         # o3d.visualization.draw_geometries([pcd,p4,p40])
    #
    # aset=np.zeros((dd-uu,rr-ll))
    # for i in range(uu, dd):
    #     for j in range(ll, rr):
    #         # XYZ.append(two2three(i,j,dp))
    #         size2=7
    #         # angel=vg.angle(np.array(normalset[i-uu][j-ll]), np.array(E[:3]))
    #         angel = vg.angle(np.array(normalset_S[i - uu][j - ll]), np.array(normalset[i - uu][j - ll]))
    #         aset[i - uu][j - ll]=min(angel,180-angel)
    #         #or angle(normalset[i-uu][j-ll],[0,0,1])<30/180*math.pi
    #         # if np.array(normalset_S[i - uu][j - ll]).all() != np.array(normalset[i - uu][j - ll]).all():
    # meana=aset.mean()
    #
    # for i in range(uu, dd):
    #     for j in range(ll, rr):
    #         ang=aset[i - uu][j - ll]
    #         if ang!=0 and (180-ang)!=0:
    #                 print(meana)
    #                 if i - size2 < uu or i + size2 > dd or j - size2 < ll or j + size2 > rr or ang<2*meana or (180-ang)<2*meana:
    #                     k1s[i][j] = 0
    #                     k2s[i][j] = 0
    #                     print("same")
    #                     continue
            # oll=sourond([i,j],size2,dp,normalset[i-uu][j-ll])
            # print(oll)
            # if oll<1.2:
            #     k1s[i][j]=0
            #     k2s[i][ j] = 0
            #     print("same")
            # kk=curvelocal([i, j],1, dp,6,uu,dd,ll,rr)

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
            # elif lbsel[i][j]==3:
            #     xyz4.append([px[i+uu][j+ll],py[i+uu][j+ll],pz[i+uu][j+ll]])
            #     tt[i + uu][j + ll] = np.array([q4[0], q4[1], q4[2]])
            #     tt4[i + uu][j + ll] = np.array([q4[0], q4[1], q4[2]])
    cv2.imwrite(rootpath+'/11.13/tt.png', tt)
    cv2.imwrite(rootpath+'/11.13/t1.png', tt1)
    cv2.imwrite(rootpath+'/11.13/t2.png', tt2)
    cv2.imwrite(rootpath+'/11.13/t3.png', tt3)
    # cv2.imwrite(rootpath+'/11.13/t4.png', tt4)
    write_pointcloud(rootpath+'/11.13/w1.ply',np.array(xyz1))
    write_pointcloud(rootpath+'/11.13/w2.ply',np.array(xyz2))
    write_pointcloud(rootpath+'/11.13/w3.ply',np.array(xyz3))
    # write_pointcloud(rootpath+'/11.13/w4.ply',np.array(xyz4))
    # tt2=image.copy()
    #

    plt.imshow(tt)

    plt.show()



    for i in pointset2:

        xx1.append(i[0])

        yy1.append(i[1])

        zz1.append(i[2])

    print(pointset2, "...............")




    plt.subplot(221)


    image = bgr_rgb(image / 255)

    plt.subplot(222)

    plt.imshow(image)

    plt.subplot(223)

    plt.imshow(dp)

    plt.subplot(224)

    plt.imshow(bgr_rgb(frame))

    # plt.show()




frame=cv2.imread(rootpath+'/11.13/36/cool2.png')
dpp=np.load(rootpath+'/11.13/36/dep1111.npy')
a = cv2.imread(rootpath+'/11.13/39/cool2.png')
b=np.load(rootpath+'/11.13/39/dep1111.npy')
points=np.load(rootpath+'/11.13/39/depcloud.npy')
print(points.shape)
see=output(a,b,frame,dpp,points)