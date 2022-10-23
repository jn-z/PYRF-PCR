PYRF-PCR
a novel three-stage outdoor point cloud registration algorithm, including preprocessing, yaw angle estimation, coarse registration, and fine registration (in short, PYRF-PCR)

1. Software:
Python and Matlab are required to implement the algorithm described in this paper:
Python version is 3.6.8
The version of Matlab is 2021b (this version and later versions of Matlab contain a point cloud processing toolbox)

2. Data set:
Link to [SemanticKITTI dataset](http://semantic-kitti.org/
The data is organized in the following format:
```
/kitti/dataset/
          └── sequences/
                  ├── 00/
                  │   ├── poses.txt
                  │   ├── image_2/
                  │   ├── image_3/
                  │   ├── labels/
                  │   │     ├ 000000.label
                  │   │     └ 000001.label
                  |   ├── voxels/
                  |   |     ├ 000000.bin
                  |   |     ├ 000000.label
                  |   |     ├ 000000.occluded
                  |   |     ├ 000000.invalid
                  |   |     ├ 000001.bin
                  |   |     ├ 000001.label
                  |   |     ├ 000001.occluded
                  |   |     ├ 000001.invalid
                  │   └── velodyne/
                  │         ├ 000000.bin
                  │         └ 000001.bin
                  ├── 01/
                  ├── 02/
                  .
                  .
                  .
                  └── 21/
```

3. Code:
The path of the program read file and output file can be modified according to the dataset or data storage location.
The sequence, name and function of main codes are as follows:
01 read_kitti.py                                         
-This code is used to read point clouds and labels from the dataset.
02 trans_semantic_to_data.m                   
-This code is used to simplify labels into four categories and convert point clouds into formats that are easy to be processed by neural networks.
03 main_SPGCN.py                                   
-This code is the SPGCN network proposed in the paper, which is used to filter the points not needed for registration from the point cloud.
04 plot_result.m                                       
-This code is used to display the point cloud filtering results.
05 Point_cloud_registration_algorithm.m 
-This code is used for point cloud registration, and has the result visualization function.
