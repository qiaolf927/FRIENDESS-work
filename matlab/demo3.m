close all
clear
clc

radius = 150;
pname = 'D:\工作\照片测试\2015-06-18 纬朗的蓝色A30D70环形光源 + Computar 50mm镜头\眼镜板-镜面十字星和圆\[DEV0]-ID015452-BW.bmp';
imo = imread(pname);
figure;imshow(imo);

OstuT = graythresh(imo);
imbw = im2bw(imo,OstuT);
imbw = ~imbw;
figure;imshow(imbw);

[L,NUM] = bwlabel(imbw,8);
figure;imshow(L)

imedge=edge(imbw,'canny',[0.05 0.12],3.0);
figure;imshow(imedge);

% [hough_space,hough_circle,para] = hough_Circle(imedge,2,pi/10,radius-10,radius+10,0.8);
% figure;imshow(hough_circle);