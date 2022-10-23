import open3d as o3d
import numpy as np
import scipy.io
import os
# Batch processing -------------------------------------------------------------------
# Read pointcloud files
# Point cloud data folder, for example, "E:\\SPGCN\\data\\00\\velodyne"
path_read_points = "E:\\SPGCN\\data\\00\\velodyne" 
# Folder of converted data, for example, "E:\\SPGCN\\data\\p00\\velodyne"
path_write_points = "E:\\SPGCN\\data\\p00\\velodyne"
# Get all file names under the folder
files_points = os.listdir(path_read_points)
# Traverse folder
for file in files_points: 
    points = []
    position_read_points = path_read_points+'\\'+ file
    points = np.fromfile(position_read_points, dtype=np.float32).reshape(-1, 4)
    position_write_points = path_write_points+'\\'+ file + '.mat'
    scipy.io.savemat(position_write_points,{'points':points})  
# Get all tag files
path_read_labels = "E:\\SPGCN\\data\\00\\labels" 
path_write_labels = "E:\\SPGCN\\data\\p00\\labels"
files_labels = os.listdir(path_read_labels) 
# Traverse folder
for file in files_labels: 
    label = []
    position_read_labels = path_read_labels+'\\'+ file
    label = np.fromfile(position_read_labels, dtype=np.uint32).reshape((-1))
    position_write_labels = path_write_labels+'\\'+ file + '.mat'
    scipy.io.savemat(position_write_labels,{'label':label})  

# Single file processing --------------------------------------------------------------------------
# For example
points = np.fromfile("E:\\SPGCN\\data\\00\\velodyne\\000010.bin", dtype=np.float32).reshape(-1, 4)
scipy.io.savemat('E:\\SPGCN\\data\\p00\\points.mat',{'points':points})  
label = np.fromfile("E:\\SPGCN\\data\\00\\labels\\000010.label", dtype=np.uint32).reshape((-1))
scipy.io.savemat('E:\\SPGCN\\data\\p00\\label.mat',{'label':label})  
# semantic label in lower half 
sem_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
sem_label = label & 0xFFFF  
scipy.io.savemat('C:\\Users\\admin\\Desktop\\labelkitti\\sigel_data\\sem_label.mat',{'sem_label':sem_label})
# instance id in upper half    
ins_label = label >> 16    
scipy.io.savemat('C:\\Users\\admin\\Desktop\\labelkitti\\sigel_data\\ins_label.mat',{'ins_label':ins_label})
# Plot pointcloud
clouds=points[:,:3]
colors=np.zeros([len(label),3])
for cla in range(len(label)):
    #print(label[cla])
    if label[cla] == 40 or label[cla] == 44 or label[cla] == 48 or label[cla] == 49 or label[cla] == 60 :
        colors[cla][0]=0
        colors[cla][1]=1
        colors[cla][2]=0
    else:
        colors[cla][0]=0
        colors[cla][1]=0
        colors[cla][2]=1
test_pcd = o3d.geometry.PointCloud()
test_pcd.points = o3d.utility.Vector3dVector(clouds)
test_pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([test_pcd], window_name="Open3D2")
