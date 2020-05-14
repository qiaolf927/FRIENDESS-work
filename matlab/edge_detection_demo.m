close all
clear
clc

pname = 'D:\工作\照片测试\2015-06-26 桌上视觉测试台\[DEV0]-ID002472-BW.bmp';
imo = imread(pname);
figure;imshow(imo);

imedge=edge(imo,'canny',[0.05 0.12],5.0);
figure;imshow(imedge);