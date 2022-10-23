clc
clear all
close all

% Convert multi category labels of original data to four categories
% Read Data
fileFolder_points = fullfile('E:\SPGCN\data\p00\velodyne');  
dirOutput_points = dir(fullfile(fileFolder_points,'*.mat')); 
fileNames_points = {dirOutput_points.name};                  
fileFolder_labels = fullfile('E:\SPGCN\data\p00\labels');     
dirOutput_labels = dir(fullfile(fileFolder_labels,'*.mat'));  
fileNames_labels = {dirOutput_labels.name};                  
onlyName = strrep(fileNames_points,'.bin.mat','');
% Storage path
write_path = fullfile('E:\SPGCN\data\p00\XYZLR');


for order_number = 1:1:length(dirOutput_points)
    write = [write_path,'\',onlyName{1,order_number},'.XYZFLR.mat'];
    % Read pointcloud data
    points_name = [fileFolder_points,'\',fileNames_points{1,order_number}];
    points = load(points_name);
    points = points.points;
    % Read label data
    labels_name = [fileFolder_labels,'\',fileNames_labels{1,order_number}];
    label = load(labels_name);
    label = label.label';
    
    % Modify labels
    % Invalid
    loc = find(label == 0 | label == 1);
    label(loc) = 0;
    % Ground
    loc = find(label == 40 | label == 44 | label == 48 | label == 49 | label == 60);
    label(loc) = 1;
    % Moving objects
    loc = find(label == 10 | label == 11 | label == 13 | label == 15 | label == 16 | label == 18 | label == 20 | label == 30 | label == 31 | label == 32 | label == 252 | label == 253 | label == 254 | label == 255 | label == 256 | label == 257 | label == 258 | label == 259 | label > 259);
    label(loc) = 2;
     % Fixed objects
    loc = find(label > 2);
    label(loc) = 3;   
    
    % Meshing Point Clouds
    A = points;
    [azimuth,pitch,R] = cart2sph(A(:,1),A(:,2),A(:,3));
    [a,b] = size(A);
    
    x = {};
    y = {};
    z = {};
    f = {};
    L = {};
    fw = {};
    fy = {};
    
    chu = 0.08 * 2; % Azimuth interval of lidar
    line_num = 1;
    azimuth(azimuth < 0) = azimuth(azimuth < 0) + 2 * pi;
    p = 1;
    for i = 1:1:(a - 1)
        if abs(azimuth(i) - azimuth(i + 1)) > pi      
            x{line_num}(p) = A(i,1);
            y{line_num}(p) = A(i,2);
            z{line_num}(p) = A(i,3);
            f{line_num}(p) = A(i,4);
            L{line_num}(p) = label(i,1);
            fw{line_num}(p) = azimuth(i,1);
            fy{line_num}(p) = pitch(i,1);
            line_num = line_num + 1;
            p = 1;                          
        else                                
            x{line_num}(p) = A(i,1);
            y{line_num}(p) = A(i,2);
            z{line_num}(p) = A(i,3);
            f{line_num}(p) = A(i,4);
            L{line_num}(p) = label(i,1);
            fw{line_num}(p) = azimuth(i,1);
            fy{line_num}(p) = pitch(i,1);
            p = p + 1;                      
        end
    end
    
    x{line_num}(p) = A(a,1);
    y{line_num}(p) = A(a,2);
    z{line_num}(p) = A(a,3);
    f{line_num}(p) = A(a,4);
    L{line_num}(p) = label(a,1);
    fw{line_num}(p) = azimuth(a,1);
    fy{line_num}(p) = pitch(a,1);
    
    fw_fin = [];
    fy_fin = [];
    x_fin = [];
    y_fin = [];
    z_fin = [];
    f_fin = [];
    label_fin = [];
    
    for j = (chu / 180 * pi):(chu / 180 * pi):2 * pi
        
        tem_fw = zeros(line_num,1);
        tem_fy = zeros(line_num,1);
        tem_x = zeros(line_num,1);
        tem_y = zeros(line_num,1);
        tem_z = zeros(line_num,1);
        tem_f = zeros(line_num,1);
        tem_L = zeros(line_num,1);
        
        for i = 1:1:line_num     
            [row col] = find(fw{1,i} < j);
            if ~isempty(row)               
                tem_fw(i,1) = fw{1,i}(row(1),col(1));
                tem_fy(i,1) = fy{1,i}(row(1),col(1));
                tem_x(i,1) = x{1,i}(row(1),col(1));
                tem_y(i,1) = y{1,i}(row(1),col(1));
                tem_z(i,1) = z{1,i}(row(1),col(1));
                tem_f(i,1) = f{1,i}(row(1),col(1));
                tem_L(i,1) = L{1,i}(row(1),col(1));
                
                fw{1,i}(fw{1,i} < j) = 2 * pi;
                
            end
        end
        if tem_fw == zeros(line_num,1);   % 全是0的情况
            fw_fin = [fw_fin,tem_fw];
            fy_fin = [fy_fin,tem_fy];
            x_fin = [x_fin,tem_x];
            y_fin = [y_fin,tem_y];
            z_fin = [z_fin,ttemem_z];
            f_fin = [f_fin,tem_f];
            label_fin = [label_fin,tem_L];
        else                             
            fw_fin = [fw_fin,tem_fw];
            fy_fin = [fy_fin,tem_fy];
            x_fin = [x_fin,tem_x];
            y_fin = [y_fin,tem_y];
            z_fin = [z_fin,tem_z];
            f_fin = [f_fin,tem_f];
            label_fin = [label_fin,tem_L];
        end
    end
    points_X = flipud(x_fin);
    points_Y = flipud(y_fin);
    points_Z = flipud(z_fin);
    points_F = flipud(f_fin);
    points_label = flipud(label_fin);
    
    [azimuth pitch R] = cart2sph(points_X,points_Y,points_Z);
    % Store the results of data processing
    save(write,'points_X','points_Y','points_Z','points_F','R','points_label')
    rate = order_number / length(dirOutput_points)  % Show program progress
end