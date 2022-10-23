clc
clear all
close all

% Set the threshold of density distribution
threshold = 0.1;

%% Read filtered point cloud and label data（Source point cloud）
A = load('000001.XYZFLR.mat');
x = A.points_X;
y = A.points_Y;
z = A.points_Z;
L = A.points_label;
R = A.R;

loc = find(R > 0);
xx = x(loc);
yy = y(loc);
zz = z(loc);
SOURCE = pointCloud([xx(:) yy(:) zz(:)]);

loc = find(L > 12);
x = x(loc);
y = y(loc);
z = z(loc);

% figure
% plot3(x,y,z,'b.','MarkerSize',0.5)
% xlabel('x')
% ylabel('y')
% zlabel('z')
% grid on
% axis equal

%% Meshing
hang = linspace(-40,40,81);
lie = linspace(-40,40,81);
 
fanwei = cell(80,80);
u = [x y];
 
N_rectan = sum( u(:,1)>=-40 & u(:,1)<40 & u(:,2)>=-40 & u(:,2)<40 );
%% Calculate distribution frequency
for i =1:length(hang)-1
    for j = 1:length(lie)-1
        index = u(:,1)>=hang(i) & u(:,1)<hang(i+1) & u(:,2)>=lie(j) & u(:,2)<lie(j+1);
        fanwei{j,i} = [hang(i) hang(i+1) lie(j) lie(j+1)];
        Z(j,i) = sum(index);
    end
end
Z_fre = Z/ N_rectan;
Z_fre = Z_fre(end:-1:1,:);
figure
imagesc(Z_fre);colorbar
Z_fre(find(Z_fre < (threshold * max(max(Z_fre))))) = 0;
Z_fre(find(Z_fre >= (threshold * max(max(Z_fre))))) = 1;
%% Frequency distribution diagram
figure
imagesc(Z_fre);colorbar


xyz = [];
Z_fre = Z_fre(end:-1:1,:);
for i = 1:1:80
    for j = 1:1:80
        if Z_fre(i,j) == 1
            loc_xx = find(x >= fanwei{i,j}(1) & x < fanwei{i,j}(2) & y >= fanwei{i,j}(3) & y < fanwei{i,j}(4));
            xyz = [xyz;x(loc_xx) y(loc_xx) z(loc_xx)];
        end
    end
end
% figure
% plot(xyz(:,1),xyz(:,2),'r.')
% grid on
% axis equal
y = xyz(:,2);
loc_zuo = find(y <= 0);
loc_you = find(y > 0);
source_zuo = pointCloud(xyz(loc_zuo,:));
source_you = pointCloud(xyz(loc_you,:));
source = pointCloud(xyz);


%% Read filtered point cloud and label data（Target point cloud）
A = load('000004.XYZFLR.mat');
x = A.points_X;
y = A.points_Y;
z = A.points_Z;
L = A.points_label;
R = A.R;

loc = find(R > 0);
xx = x(loc);
yy = y(loc);
zz = z(loc);
TARGET = pointCloud([xx(:) yy(:) zz(:)]);

loc = find(L > 12);
x = x(loc);
y = y(loc);
z = z(loc);


hang = linspace(-40,40,81);
lie = linspace(-40,40,81);

fanwei = cell(80,80);
u = [x y];

N_rectan = sum( u(:,1)>=-40 & u(:,1)<40 & u(:,2)>=-40 & u(:,2)<40 );

for i =1:length(hang)-1
    for j = 1:length(lie)-1
        index = u(:,1)>=hang(i) & u(:,1)<hang(i+1) & u(:,2)>=lie(j) & u(:,2)<lie(j+1);
        fanwei{j,i} = [hang(i) hang(i+1) lie(j) lie(j+1)];
        Z(j,i) = sum(index);
    end
end
Z_fre = Z/ N_rectan;
Z_fre = Z_fre(end:-1:1,:);
figure
imagesc(Z_fre);colorbar
Z_fre(find(Z_fre < (threshold * max(max(Z_fre))))) = 0;
Z_fre(find(Z_fre >= (threshold * max(max(Z_fre))))) = 1;

figure
imagesc(Z_fre);colorbar

xyz = [];
Z_fre = Z_fre(end:-1:1,:);
for i = 1:1:80
    for j = 1:1:80
        if Z_fre(i,j) == 1
            loc_xx = find(x >= fanwei{i,j}(1) & x < fanwei{i,j}(2) & y >= fanwei{i,j}(3) & y < fanwei{i,j}(4));
            xyz = [xyz;x(loc_xx) y(loc_xx) z(loc_xx)];
        end
    end
end
% figure
% plot(xyz(:,1),xyz(:,2),'r.')
% grid on
% axis equal
y = xyz(:,2);
loc_zuo = find(y <= 0);
loc_you = find(y > 0);
target_zuo = pointCloud(xyz(loc_zuo,:));
target_you = pointCloud(xyz(loc_you,:));
target = pointCloud(xyz);


%% Improved FPFH
% Visualize point cloud
figure
pcshowpair(source,target);
legend("Original", "Transformed","TextColor",[1 1 1]);
 
fixedFeature = extractFPFHFeatures(target);
movingFeature = extractFPFHFeatures(source);
length(movingFeature)

fixedFeature_zuo = extractFPFHFeatures(target_zuo);
movingFeature_zuo = extractFPFHFeatures(source_zuo);

fixedFeature_you = extractFPFHFeatures(target_you);
movingFeature_you = extractFPFHFeatures(source_you);

[matchingPairs,scores] = pcmatchfeatures(fixedFeature,movingFeature,target,source);
length(matchingPairs)
tic
[matchingPairs_zuo,scores_zuo] = pcmatchfeatures(fixedFeature_zuo,movingFeature_zuo,target_zuo,source_zuo);
[matchingPairs_you,scores_you] = pcmatchfeatures(fixedFeature_you,movingFeature_you,target_you,source_you);
toc
 
mean(scores)
 
matchedPts1 = select(target,matchingPairs(:,1));
matchedPts2 = select(source,matchingPairs(:,2));

matchedPts1_zuo = select(target_zuo,matchingPairs_zuo(:,1));
matchedPts2_zuo = select(source_zuo,matchingPairs_zuo(:,2));

matchedPts1_you = select(target_you,matchingPairs_you(:,1));
matchedPts2_you = select(source_you,matchingPairs_you(:,2));


matchedPts11 = [matchedPts1_zuo.Location;matchedPts1_you.Location];
matchedPts22 = [matchedPts2_zuo.Location;matchedPts2_you.Location];

matchedPts11 = pointCloud(matchedPts11);
matchedPts22 = pointCloud(matchedPts22);

figure
pcshowMatchedFeatures(target,source,matchedPts11,matchedPts22, ...
    "Method","montage")

tic
estimatedTform = estimateGeometricTransform3D(matchedPts11.Location, ...
    matchedPts22.Location,"rigid");
toc
disp(estimatedTform.T)
 
source = pctransform(source,invert(estimatedTform));
matchedPts3 = pctransform(matchedPts2,invert(estimatedTform));
matchedPts33 = pctransform(matchedPts22,invert(estimatedTform));

SOURCE_new = pctransform(SOURCE,invert(estimatedTform));

% figure
% pcshowpair(matchedPts2,matchedPts1);

A = matchedPts11.Location;
B = matchedPts33.Location;
[chang kuan] = size(A);
wrong_loc = [];
right_loc = [];
for i = 1:1:length(A)
    tem = A(i,:) - B(i,:);
    tem = sqrt(tem(1)^2 + tem(2)^2 + tem(3)^2);
    if tem > 1
        wrong_loc = [wrong_loc,i];
    else
        right_loc = [right_loc,i];
    end
end
% save('wrong_loc.mat','wrong_loc')
% figure
% pcshowMatchedFeatures2(target,source,matchedPts11,matchedPts22, ...
%     "Method","montage")

S = source;
P = S.Location;
xmin1 = min(P(:,1));
xmax1 = max(P(:,1));
matche_S = select(S,matchingPairs(:,2));
matche_P = matche_S.Location;

T = target;
Q = T.Location;
xmin2 = min(Q(:,1));
xmax2 = max(Q(:,1));
matche_T = select(T,matchingPairs(:,1));
matche_Q = matche_T.Location;

x_cha = abs(xmax1 - xmin2);
Q(:,1) = Q(:,1) + x_cha;
matche_Q(:,1) = matche_Q(:,1) + x_cha;

lineX_r = [matche_P(right_loc,1)'; matche_Q(right_loc,1)'];
numPts_r = numel(lineX_r);
lineX_r = [lineX_r; NaN(1,numPts_r/2)];
lineY_r = [matche_P(right_loc,2)'; matche_Q(right_loc,2)'];
lineY_r = [lineY_r; NaN(1,numPts_r/2)];
lineZ_r = [matche_P(right_loc,3)'; matche_Q(right_loc,3)'];
lineZ_r = [lineZ_r; NaN(1,numPts_r/2)];

lineX_w = [matche_P(wrong_loc,1)'; matche_Q(wrong_loc,1)'];
numPts_w = numel(lineX_w);
lineX_w = [lineX_w; NaN(1,numPts_w/2)];
lineY_w = [matche_P(wrong_loc,2)'; matche_Q(wrong_loc,2)'];
lineY_w = [lineY_w; NaN(1,numPts_w/2)];
lineZ_w = [matche_P(wrong_loc,3)'; matche_Q(wrong_loc,3)'];
lineZ_w = [lineZ_w; NaN(1,numPts_w/2)];

figure
plot3(P(:,1),P(:,2),P(:,3),'b.','MarkerSize',0.1)
hold on
plot3(Q(:,1),Q(:,2),Q(:,3),'r.','MarkerSize',0.1)
hold on
plot3(lineX_r(:), lineY_r(:), lineZ_r(:), 'm-','LineWidth',1); % line
hold on
plot3(lineX_w(:), lineY_w(:), lineZ_w(:), 'k-','LineWidth',0.1); % line
axis equal
grid on
legend('Source point cloud','Target point cloud','Correct pairing','Wrong pairing')

figure
pcshowpair(target,source)
figure
plot(target.Location(1:2:end,1),target.Location(1:2:end,2),'b.','MarkerSize',0.1)
hold on
plot(source.Location(1:2:end,1),source.Location(1:2:end,2),'r.','MarkerSize',0.1)
grid on
axis equal
axis([-25 25 -25 25])
title('Point cloud after registration')

figure;
pcshowpair(TARGET,SOURCE);
figure
plot(TARGET.Location(1:2:end,1),TARGET.Location(1:2:end,2),'b.','MarkerSize',0.1)
hold on
plot(SOURCE.Location(1:2:end,1),SOURCE.Location(1:2:end,2),'r.','MarkerSize',0.1)
grid on
axis equal
axis([-25 25 -25 25])
title('Point cloud before registration')

% xlim([-50 50])
% ylim([-40 60])
% title("Aligned Point Clouds")
figure;
pcshowpair(TARGET,SOURCE_new);
figure
pcshowpair(target,source)

figure
plot(TARGET.Location(1:2:end,1),TARGET.Location(1:2:end,2),'b.','MarkerSize',0.1)
hold on
plot(SOURCE_new.Location(1:2:end,1),SOURCE_new.Location(1:2:end,2),'r.','MarkerSize',0.1)
grid on
axis equal
axis([-25 25 -25 25])
title('Point cloud after registration')
