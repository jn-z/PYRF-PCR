clc
clear all
close all
% Draw ground filtering results

% ground filtering results
label_result = load('E:\SPGCN\data\p00\result2\result_000004.mat');
label_result = label_result.predict_write;

% Point cloud data and original labels
XYZLR = load('E:\SPGCN\data\p00\XYZLR\000004.XYZFLR.mat');

X = XYZLR.points_X;
Y = XYZLR.points_Y;
Z = XYZLR.points_Z;

label_points = XYZLR.points_label;

% Ground points and non ground points of the original label file
loc_p_0 = find(label_points == 0);  % Invalid point
loc_p_1 = find(label_points == 1);  % Ground point
loc_p_2 = find(label_points == 2);  % Moving object point
loc_p_3 = find(label_points == 3);  % Fixed object point

figure
plot3(X(loc_p_1),Y(loc_p_1),Z(loc_p_1),'g.','MarkerSize',0.5)
hold on
plot3(X(loc_p_2),Y(loc_p_2),Z(loc_p_2),'r.','MarkerSize',0.5)
hold on
plot3(X(loc_p_3),Y(loc_p_3),Z(loc_p_3),'b.','MarkerSize',0.5)
grid on
axis equal
view([-1,1,1]);
axis([-60 60 -60 60])

% Ground points and non ground points in the result
loc_r_0 = find(label_result == 0);  
loc_r_1 = find(label_result == 1);  
loc_r_2 = find(label_result == 2);  
loc_r_3 = find(label_result == 3);  

figure
plot3(X(loc_r_1),Y(loc_r_1),Z(loc_r_1),'g.','MarkerSize',0.5)
hold on
plot3(X(loc_r_2),Y(loc_r_2),Z(loc_r_2),'r.','MarkerSize',0.5)
hold on
plot3(X(loc_r_3),Y(loc_r_3),Z(loc_r_3),'b.','MarkerSize',0.5)
grid on
axis equal
view([-1,1,1]);

