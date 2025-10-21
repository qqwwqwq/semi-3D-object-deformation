"""
PERIODIC DYNAMIC MOVEMENT PRIMITIVES (pDMP)

An example of how to use pDMP functions.


AUTHOR: Luka Peternel
e-mail: l.peternel@tudelft.nl


REFERENCE:
L. Peternel, T. Noda, T. Petrič, A. Ude, J. Morimoto and J. Babič
Adaptive control of exoskeleton robots for periodic assistive behaviours based on EMG feedback minimisation,
PLOS One 11(2): e0148942, Feb 2016

"""
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
rootpath="/mnt/storage/buildwin/desk_backword"
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
    color=["purple", "black","blue"]
    plt.xlim(-90, 40)
    plt.ylim(-60, 70)

    from scipy.spatial import distance
    for j in range(len(threepset)):
        d = []
        threedline = threepset[j][:, 3]
        center = threepset[j][:, 0]
        rp = threepset[j][:, 1]
        lp = threepset[j][:, 2]
        for k in range(rp.shape[0]):
            d.append(distance.euclidean(rp[k],lp[k]))
        ax.scatter3D(rp[:, 0], rp[:, 1], rp[:, 2], marker='^')
        ax.scatter3D(lp[:, 0], lp[:, 1], lp[:, 2],marker='.')
        # for i in range(0, len(center)):
        #     ax.plot3D(np.array([rp[i][0], lp[i][0]]), np.array([rp[i][1], lp[i][1]]), np.array([rp[i][2], lp[i][2]]),
        #               c=color2[j])
        ax.plot3D(threedline[:, 0], threedline[:, 1], threedline[:, 2], color[j])
        print(min( d),max(d), "---")

    plt.show()
nm=49
threepset = np.load(rootpath+"/11.13/" + str(nm) + "/0_threepset_n.npy",allow_pickle = True)
threepset2 = np.load(rootpath+"/11.13/" + str(nm) + "/1_threepset_n.npy",allow_pickle = True)
# threepset3 = np.load(rootpath+"/11.13/" + str(nm) + "/2_threepset_n.npy",allow_pickle = True)
origin([threepset,threepset2])
# threedline=np.load(rootpath+"/11.13/1_threedline.npy",allow_pickle = True)
E= np.load(rootpath+"/11.13/" + str(nm) + "/0_plane_n.npy",allow_pickle = True)
E2= np.load(rootpath+"/11.13/" + str(nm) + "/1_plane_n.npy",allow_pickle = True)
# E3= np.load(rootpath+"/11.13/" + str(nm) + "/2_plane_n.npy",allow_pickle = True)
# threedline=threepset[:,3]
# center=threepset[:,0]
# setsamples=len(center)
# print(setsamples,1111)
# rp=threepset[:,1]
# lp=threepset[:,2]
def main(threepset,E):
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # color2 = ["purple", "black","yellow"]
    # plt.xlim(-90, 40)
    # plt.ylim(-60, 70)
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
            a = dmtracject(20, 0.1, dis)
            a.generate()
            a.show()
            b = gaindmp(20, 0.1, dis, a.data[:, 1])
            # b.show()
            t, y = b.update(5)
            b.show()
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





            ax.plot3D(threedline[:, 0], threedline[:, 1], threedline[:, 2], 'gray')
            ax.scatter3D(rp[:, 0], rp[:, 1], rp[:, 2], marker='^')
            ax.scatter3D(lp[:, 0], lp[:, 1], lp[:, 2], marker='.')
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

main([threepset,threepset2],[E,E2])
    # cross_product = np.cross(unit_v1, unit_v2)
    # if cross_product.item() > 0:
    #     angle = 2 * np.pi - angle



# ax.scatter3D(center[:,0],center[:,1],center[:,2], cmap='Blues')
#绘制空间曲线






# def corss(lp, rp, stp, edp):
#     v1 = lp - stp
#     v2 = rp - stp
#     v3 = edp - stp
#     v4=lp-rp
#     v5=stp-rp
#     v6=edp-rp
#     a3=np.cross(v5,v4)
#     a4=np.cross(v6,v4)
#     a1 = np.cross(v1, v3)
#     a2 = np.cross(v2, v3)
#     if a1 * a2 > 0 or a3*a4>0:
#         return False
#     return True
# for i in range(2,setsamples-2):
#     stp = center[i - 2]
#     edp = center[i + 2]
#     rp1 = rp[i]
#     lp1 = lp[i]
#     rm=rotation_matrix_from_vectors(np.array([0,0,1]),E[:3])
#     stp=np.dot(stp,rm)
#     edp=np.dot(edp,rm)
#     rp1=np.dot(rp1,rm)
#     lp1=np.dot(lp1,rm)
#     print(corss(lp1[:2],rp1[:2],stp[:2],edp[:2]))
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.plot3D(np.array([rp1[0],lp1[0]]), np.array([rp1[1],lp1[1]]), np.array([rp1[2],lp1[2]]), 'red')
#     ax.plot3D(np.array([stp[0], edp[0]]), np.array([stp[1], edp[1]]), np.array([stp[2], edp[2]]), 'blue')
#     plt.show()

    # if i>20:
    #
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     ax.plot3D(np.array(resultx), np.array(resulty), np.array(resultz), 'red')
    #     ax.plot3D(threedline[:, 0][0:i+3], np.array(threedline[:, 1][0:i+3]), np.array(threedline[:, 2][0:i+3]), 'blue')
    #     # ax.plot3D(center[:, 0], center[:, 1], center[:, 2], 'gray')
    # # ax.scatter3D([rp1[0],lp1[0]], [rp1[1],lp1[1]], [rp1[2],lp1[2]], cmap='blue')
    # # ax.plot3D(np.asarray(rp[i+1][0],lp[i+1][0]), np.asarray(lp[i+1][1],rp[i+1][1]), np.asarray(lp[i+1][2],rp[i+1][2]), cmap='blue')
    #     plt.show()





    # ax.plot3D(after_rt2[:, 0], after_rt2[:, 1], after_rt2[:, 2], 'red')
    # # ax.plot3D(after_rt[:, 0], after_rt[:, 1], after_rt[:, 2], 'red')
    # ax.scatter3D(np.array([p1[0],p2[0]]), np.array([p1[1],p2[1]]), np.array([p1[2],p2[2]]), cmap='Blues')
    # ax.scatter3D(rp[:, 0], rp[:, 1], rp[:, 2], cmap='red')
    # ax.scatter3D(lp[:, 0], lp[:, 1], lp[:, 2], cmap='red')
    # ax.plot3D(center[:, 0], center[:, 1], center[:, 2], 'gray')
    # plt.plot(after_rt[:, 0], after_rt[:, 1], "b", label="adapted value")
    # # plt.plot(welement[:, 0], welement[:, 1], "r", label="adapted value")
    # plt.plot([p1[0],p2[0],], [p1[1],p2[1]], "r", label="adapted value")
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.show()
    # b.show()




# # EXPERIMENT PARAMETERS
# dt = 0.1 # system sample time
# exp_time = 80 # total experiment time
# samples = int(1/dt) * exp_time
# DOF = 4 # degrees of freedom (number of DMPs to be learned)
# N = 25 # number of weights per DMP (more weights can reproduce more complicated shapes)
# alpha = 8 # DMP gain alpha
# beta = 2 # DMP gain beta
# lambd = 0.995 # forgetting factor
# tau = 5 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
# phi = 0 # DMP phase
#
# mode = 2 # DMP mode of operation (see below for details)
#
# y_old = 0
# dy_old = 0
#
# data = []
#
# # create a DMP object
# myDMP = pDMP(DOF, N, alpha, beta, lambd, dt)
#
#
#
# print(samples)
# # MAIN LOOP
# set=np.array([0]*samples)
# set[0:400]-=100
# set[400:]+=100
# b=np.load("filename.npy")
# for i in range ( samples ):
#
#     # generate phase
#     phi += 2*np.pi * dt/tau
#
#     # generate an example trajectory (e.g., the movement that is to be learned)
#     y = np.array([np.sin(phi), np.cos(phi), -np.sin(phi), -np.cos(phi)])
#     # calculate time derivatives
#     dy = (y - y_old) / dt
#     ddy = (dy - dy_old) / dt
#     # generate an example update (e.g., EMG singals that update exoskeleton joint torques as in [Peternel, 2016])
#     U = set[i]*y # typically update factor is an input signal multiplied by a gain
#     # set phase and period for DMPs
#     myDMP.set_phase( np.array([phi,phi,phi,phi]) )
#     myDMP.set_period( np.array([tau,tau,tau,tau]) )
#
#     print(myDMP.f)
#     # DMP mode of operation
#     if i < int( 1 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
#         if( mode == 1 ):
#             myDMP.learn(y, dy, ddy) # learn DMP based on a trajectory
#         elif ( mode == 2 ):
#             myDMP.update(U) # update DMP based on an update factor
#     else:
#         myDMP.repeat() # repeat the learned DMP
#
#     # DMP integration
#     myDMP.integration()
#     # old values
#     y_old = y
#     dy_old = dy
#     # store data for plotting
#     x, dx, ph, ta = myDMP.get_state()
#     time = dt*i
#     data.append([time,phi,x[3],y[3]])
# # PLOTS
# data = np.asarray(data)
# # input
# plt.plot(data[:,0][::25],data[:,3][::25],'r')
# # DMP
# plt.plot(data[:,0],data[:,2],'b')
# # np.save("filename.npy",data[:,2])
# o=np.pi/12
# matrix=np.array([[math.cos(o),-math.sin(o)],[math.sin(o),math.cos(o)]])
#
# ts=np.array([data[:,0],data[:,2]])
#
# d=np.dot(matrix,ts)
# # plt.plot(d[0],d[1],'r')
# # print(matrix,ts,d)
# # plt.plot(ts[0],d[0],'y')
# # plt.plot(ts[0],d[1],'y')
# plt.xlabel('time [s]', fontsize='12')
# plt.ylabel('signal', fontsize='13')
# ax = plt.gca()
# # ax.set_aspect(1)
# plt.legend(['input','DMP'])
# plt.title('Periodic DMP', fontsize='14')
# plt.show()







