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
rootpath = "/mnt/storage/buildwin/desk_backword"


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
    o3d.visualization.draw_geometries([pcd__])
    o3d.visualization.draw_geometries([widout])
    # 2d curve
    plano1 = nLine()
    a1s=[120]
    # a2s = [ 30, 45, 60, 75,90]
    for k in a1s:
        A, best_inliers, pt = plano1.fit(np.asarray(pcd__.points), E, maxIteration=20000, tre2=1.5,a1=k,a2=k)
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
        o3d.visualization.draw_geometries([win, ts])
        o3d.visualization.draw_geometries([widout, ts])
        # o3d.visualization.draw_geometries([pcd__, ts])
        # o3d.visualization.draw_geometries([ptt, ts])
        # not_plane2 = pcd__.select_by_index(best_inliers, invert=True).paint_uniform_color(randomcolor())
        o3d.visualization.draw_geometries([ptt, AA, p3, ts])


    # 3d curve
    ransc = ransacspline()
    print(np.asarray(plane2.points).shape,1111111111111)
    depline, depid = ransc.fit(pts=np.asarray(plane2.points), thresh=2, maxIteration=2000)
    p = plane2.select_by_index(depid).paint_uniform_color(randomcolor())
    AA = o3d.geometry.PointCloud()
    AA.points = o3d.utility.Vector3dVector(depline)
    c=randomcolor()
    AA.paint_uniform_color(c)
    o3d.visualization.draw_geometries([p, ts])
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
    o3d.visualization.draw_geometries([widout, p2d])
    o3d.visualization.draw_geometries([ts, AA])
    o3d.visualization.draw_geometries([ts, p2d])
    nlist=[10,20,40,80,160]
    angl=[0,15,30,45,60,75]
    for k in angl:
        best, threepset = plano2.fit(widout, np.asarray(widout.points), twodcurve, threedcurve,80,k)
        print(len(best))
        wid_part = widout.select_by_index(best).paint_uniform_color(randomcolor())
        wid_not = widout.select_by_index(best, invert=True).paint_uniform_color(randomcolor())
        o3d.visualization.draw_geometries([ts, wid_part, p2d])
        # o3d.visualization.draw_geometries([wid_part, wid_not])
    remian = pcd__.select_by_index(
        remove(np.asarray(threepset)[:, 1:3], np.asarray(threepset)[:, 0], np.array(pcd__.points), 12,E),
        invert=True).paint_uniform_color(
        randomcolor())
    print(np.array(pcd__.points).shape)

    o3d.visualization.draw_geometries([ts, AA])
    o3d.visualization.draw_geometries([ts, wid_part, p2d, AA])
    o3d.visualization.draw_geometries([ pcd__,remian])
    return AA, wid_part, remian, threepset, np.asarray(AA.points)


# pcd = o3d.io.read_point_cloud(rootpath+"/11.13/w4.ply")
nm = 49
pcd2 = o3d.io.read_point_cloud(rootpath + "/11.13/" + str(nm) + "/w2.ply")
# pcd3 = o3d.io.read_point_cloud(rootpath+"/11.13/w3.ply")
pcd4 = o3d.io.read_point_cloud(rootpath + "/11.13/" + str(nm) + "/w1.ply")
# pcd2.paint_uniform_color([0, 0, 1])
# # pcd4.paint_uniform_color([0, 0, 1])
# o3d.visualization.draw_geometries([pcd2])
# pcd=pcd.voxel_down_sample(voxel_size=0.05)
# pcd2=pcd2.voxel_down_sample(voxel_size=0.05)
# pcd3=pcd3.voxel_down_sample(voxel_size=0.05)
# pcd4=pcd4.voxel_down_sample(voxel_size=0.05)
# o3d.visualization.draw_geometries([pcd,pcd3,pcd2],)
# voxel_down_pcd = pcd
plano1 = pyrsc.Plane()
E, B = plano1.fit(np.asarray(pcd4.points), thresh=3, minPoints=1000, maxIteration=1000)
print(1111111111)
p40 = pcd4.select_by_index(B).paint_uniform_color([0, 1, 0])
p4 = pcd4.select_by_index(B, invert=True).paint_uniform_color([0, 0, 1])
iner = extra(np.asarray(p4.points), E)
pout = p4.select_by_index(iner).paint_uniform_color([1, 0, 1])
pin = p4.select_by_index(iner, invert=True).paint_uniform_color([0, 1, 0])
if np.asarray(pin.points).shape[0] < np.asarray(pout.points).shape[0]:
    pout, pin = pin, pout
o3d.visualization.draw_geometries([pin])
# cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=50,
#                                                         std_ratio=0.5)
cl2, ind2 = pcd2.remove_statistical_outlier(nb_neighbors=50,
                                            std_ratio=2)
# cl3, ind3 = pcd3.remove_statistical_outlier(nb_neighbors=50,
#                                                         std_ratio=0.5)
cl4, ind4 = pin.remove_statistical_outlier(nb_neighbors=50,
                                           std_ratio=0.5)
# a,ao=display_inlier_outlier(voxel_down_pcd, ind)
b, bo = display_inlier_outlier(pcd2, ind2)
# c,co=display_inlier_outlier(pcd3, ind3)
d, do = display_inlier_outlier(pin, ind4)

# a.paint_uniform_color([0, 0, 1])
d.paint_uniform_color([1, 0.7, 0.1])
# c.paint_uniform_color([0, 1, 0])
# d.paint_uniform_color([0.5, 0, 0.5])

o3d.visualization.draw_geometries([d],
                                  )
# p3_load = np.concatenate((np.asarray(b.points),np.asarray(c.points)), axis=0)

# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
write_pointcloud(rootpath + "/11.13/" + str(nm) + "/wt.ply", np.array(b.points))
write_pointcloud(rootpath + "/11.13/" + str(nm) + "/wt2.ply", np.array(d.points))
pcd_load = o3d.io.read_point_cloud(rootpath + "/11.13/" + str(nm) + "/wt.ply")
pcd_p = o3d.io.read_point_cloud(rootpath + "/11.13/" + str(nm) + "/wt2.ply")
# pcd.paint_uniform_color([1, 0, 0])
pcd2.paint_uniform_color([0, 1, 0])
# pcd3.paint_uniform_color([0, 0, 1])zwz
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
o3d.visualization.draw_geometries([widout])
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
it = 3
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

# --------------------------------------------
# pp=np.asarray(plane2.points)
# a_arg = np.argsort(pp[:,0])
# pp = pp[a_arg].tolist()
# spline=ransacspline()
# curve,AT=spline.fit(np.asarray(pp),thresh=3, maxIteration=500)
# pc = o3d.geometry.PointCloud()
# pc.points = o3d.utility.Vector3dVector(curve)
# pc.paint_uniform_color([1, 0, 0])
# o3d.visualization.draw_geometries([pc,not_plane2])
##############
#
# # p3=plane2.select_by_index(AT,invert=False).paint_uniform_color([1, 0, 0])
# # np3=plane2.select_by_index(AT, invert=True).paint_uniform_color([0, 1, 0])
# # o3d.visualization.draw_geometries([p3,np3])
# xmax=np.max(pp,axis=0)[0]
# xmin=np.min(pp,axis=0)[0]
# ymax=np.max(pp,axis=0)[1]
# ymin=np.min(pp,axis=0)[1]
#
# finalp1.extend(findsegline(A,B,xmax,xmin,ymax,ymin))
# o3d.visualization.draw_geometries([plane2,not_plane2])
#
# A,B,best_inliers,C=plano3.fit(points,A,B, thresh=2, maxIteration=1000)
# moved_1=remove(A,B,points,thresh=5)
# moved_2=remove(A,C,points,thresh=5)
# moved_1=np.concatenate((moved_1, moved_2))
# remain_1=pcd_load.select_by_index(moved_1, invert=True).paint_uniform_color([0, 0, 1])
# plane = pcd_load.select_by_index(best_inliers).paint_uniform_color([1, 0, 0])
# not_plane = pcd_load.select_by_index(best_inliers, invert=True).paint_uniform_color([0, 0, 1])
# o3d.visualization.draw_geometries([plane,not_plane])
# # # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
#
# finalp.extend(findsegline(A,B,xmax,xmin,ymax,ymin))
# finalp.extend(findsegline(A,C,xmax,xmin,ymax,ymin))
# print(finalp,11111)
# pcd.points = o3d.utility.Vector3dVector(xyz)
# ts=2
# while ts>0:
#     print(ts)
#     pcd_load=remain_1
#     pcd_p=remian
#     points=np.asarray(pcd_load.points)
#     dp=np.asarray(pcd_p.points)
#     A, B, best_inliers = plano2.fit(dp, thresh=2, maxIteration=500)
#     moved = remove(A, B, dp, thresh=5)
#     remian = pcd_p.select_by_index(moved, invert=True).paint_uniform_color([0, 1, 0])
#     plane2 = pcd_p.select_by_index(best_inliers).paint_uniform_color([1, 0, 0])
#     not_plane2 = pcd_p.select_by_index(best_inliers, invert=True).paint_uniform_color([0, 1, 0])
#     pp = np.asarray(plane2.points)
#     xmax = np.max(pp, axis=0)[0]
#     xmin = np.min(pp, axis=0)[0]
#     ymax = np.max(pp, axis=0)[1]
#     ymin = np.min(pp, axis=0)[1]
#     finalp1.extend(findsegline(A, B, xmax, xmin,ymax,ymin))
#     o3d.visualization.draw_geometries([plane2, not_plane2])
#
#     A, B, best_inliers, C = plano3.fit(points, A, B, thresh=2, maxIteration=1000)
#     moved_1 = remove(A, B, points, thresh=5)
#     moved_2 = remove(A, C, points, thresh=5)
#     moved_1 = np.concatenate((moved_1, moved_2))
#     plane = pcd_load.select_by_index(best_inliers).paint_uniform_color([1, 0, 0])
#     not_plane = pcd_load.select_by_index(best_inliers, invert=True).paint_uniform_color([0, 0, 1])
#     o3d.visualization.draw_geometries([plane, not_plane])
#     # # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
#     finalp.extend(findsegline(A, B, xmax, xmin,ymax,ymin))
#     finalp.extend(findsegline(A, C, xmax, xmin,ymax,ymin))
#     ts-=1
#
# lines=[[0+i,1+i] for i in range(0,2*len(finalp)-1,2)]
# colors = [[1, 0, 0] for i in range(len(finalp))]
# line_set = o3d.geometry.LineSet()
# lines2=[[0+i,1+i] for i in range(0,2*len(finalp1)-1,2)]
# colors2 = [[0, 0, 1] for i in range(len(finalp1))]
# line_set2 = o3d.geometry.LineSet()
# print(lines,lines2)
# line_set2.points = o3d.utility.Vector3dVector(finalp1)
# line_set2.lines = o3d.utility.Vector2iVector(lines2)
# line_set2.colors = o3d.utility.Vector3dVector(colors2)
# line_set.points = o3d.utility.Vector3dVector(finalp)
# line_set.lines = o3d.utility.Vector2iVector(lines)
# line_set.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([line_set,line_set2])
#

# print("Downsample the point cloud with a voxel of 0.02")
# voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
# o3d.visualization.draw_geometries([voxel_down_pcd],)
# print("Every 5th points are selected")
# uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
# o3d.visualization.draw_geometries([uni_down_pcd],
#
#                                   )
