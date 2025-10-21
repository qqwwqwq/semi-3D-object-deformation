import cv2
import numpy as np
import pyransac3d as pyrsc
import pylab
import open3d as o3d
import struct
from matplotlib import pyplot as plt
from sandscan import two2three,findsand
from mpl_toolkits.mplot3d import Axes3D
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
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                     )
    return inlier_cloud,outlier_cloud
def extra( pts,Pe):
    tre=        ( Pe[0] * 0 + Pe[1] *0 + Pe[2] * 0 + Pe[3]
            ) / np.sqrt(Pe[0] ** 2 + Pe[1] ** 2 + Pe[2] ** 2)
    print(tre)
    dist_pt = (Pe[0] * pts[:, 0] + Pe[1] * pts[:, 1] + Pe[2] * pts[:, 2] + Pe[3]
              ) / np.sqrt(Pe[0] ** 2 + Pe[1] ** 2 + Pe[2] ** 2)
    print(dist_pt)
    pt_id_inliers = np.where(dist_pt <-3)[0]
    print(pt_id_inliers)
    return pt_id_inliers

rootpath="/mnt/storage/buildwin/desk_backword/11.13/evdata/6/h5/"
image = cv2.imread(rootpath+'cool2.png')
dp=np.load(rootpath+'dep1111.npy')
points=np.load(rootpath+'depcloud.npy')
pointss = []
for i in range(len(dp)):
    for j in range(len(dp[0])):
        pointss.append(two2three(i, j, dp))
si = image.copy()
sid = dp.copy()
four=findsand(si)
cut = np.load(rootpath + 'cut.npy')
print(cut)
ll=cut[0][0]
rr=cut[0][1]

uu=cut[1][0]
dd=cut[1][1]
print(uu,dd,ll,rr)
# g1 = four[0]
# b1 = four[1]
# y1 = four[2]
# r1 = four[3]
# if len(g1) == 0 or len(b1) == 0 or len(y1) == 0 or len(r1) == 0:
#     print("reference point missed-tar")
# gap=95
# ll = int(min(g1[0], b1[0], y1[0], r1[0]))+gap
# rr = int(max(g1[0], b1[0], y1[0], r1[0]))-gap
# uu = int(min(g1[1], b1[1], y1[1], r1[1]))+50
# dd = int(max(g1[1], b1[1], y1[1], r1[1]))-50
cv2.rectangle(image,(ll,uu),(rr,dd),(0,0,255),3,8)
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
fig=pylab.figure()
ax=Axes3D(fig)
ax.plot_surface(sxx,syy,szz)
# pylab.show()
sxx=sxx.flatten()
syy=syy.flatten()
szz=szz.flatten()
d=np.vstack((sxx,syy))
o=np.vstack((d,szz)).T
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(o)
plano1 = pyrsc.Plane()
E,B=plano1.fit(np.asarray(pcd.points), thresh=3, maxIteration=2000)

p40 = pcd.select_by_index(B).paint_uniform_color([0, 1, 0])
p4 = pcd.select_by_index(B, invert=True).paint_uniform_color([0, 0, 1])
o3d.visualization.draw_geometries([p40,p4])
iner=extra(np.asarray(p4.points),E)
pin=p4.select_by_index(iner).paint_uniform_color([0, 0, 1])
pout=p4.select_by_index(iner, invert=True).paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([pin,pout])
cl2, ind2 = pin.remove_statistical_outlier(nb_neighbors=200,
                                                        std_ratio=1)
b,bo=display_inlier_outlier(pin, ind2)
write_pointcloud(rootpath+'rs.ply',np.array(b.points))