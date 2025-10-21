# Importing libraries
import open3d as o3d
import numpy as np
import math
import scipy.linalg as linalg
import os
import sys
def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix
axis_x, axis_y, axis_z = [1,0,0], [0,1,0], [0, 0, 1]
rand_axis = [1,0,0]

rot_matrix= rotate_mat(axis_y, math.pi)
# Visualize point cloud
print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud( '/mnt/storage/buildwin/desk_backword/11.13/ts.ply')
p=np.asarray(pcd.points)
for i in range(p.shape[0]):
    p[i]=np.dot(rot_matrix,p[i])

pcd.points= o3d.utility.Vector3dVector(p)
print(np.asarray(pcd.points))
pcd=pcd.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([pcd])
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100))
# Voxel downsampling
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.005)
# o3d.visualization.draw_geometries([downpcd])
print(downpcd)


# Point cloud outlier removal
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


# cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# display_inlier_outlier(downpcd, ind)

# Vertex normal estimation
print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

print("Print a normal vector of the 0th point")
print(downpcd.normals[0])

print("Print the normal vectors of the first 10 points")
print(np.asarray(downpcd.normals)[:10, :])

# # Bounding volumes
# aabb = downpcd.get_axis_aligned_bounding_box()
# aabb.color = (1, 0, 0)
# # obb = downpcd.get_oriented_bounding_box()
# # obb.color = (0, 1, 0)
# o3d.visualization.draw_geometries([downpcd, aabb])
#
# # The ball pivoting mesh contruction algorithm
# radii = [0.05, 0.1, 0.2, 0.4]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     downpcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([downpcd, rec_mesh])

# # Poisson surface reconstruction
print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
print(mesh)
o3d.visualization.draw_geometries([mesh])
# # Save mesh as .STL
poisson_mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)
poisson_mesh=poisson_mesh.paint_uniform_color([1, 1, 0])
o3d.visualization.draw_geometries([poisson_mesh], mesh_show_back_face=True)
o3d.io.write_triangle_mesh("poisson_mesh_will.stl", poisson_mesh)