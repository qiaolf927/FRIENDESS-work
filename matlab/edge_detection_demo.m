close all
clear
clc

pname = 'D:\����\��Ƭ����\2015-06-26 �����Ӿ�����̨\[DEV0]-ID002472-BW.bmp';
imo = imread(pname);
figure;imshow(imo);

imedge=edge(imo,'canny',[0.05 0.12],5.0);
figure;imshow(imedge);