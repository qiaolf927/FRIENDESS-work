close all
clear
clc

pname = 'D:\工作\照片测试\2015-06-18 纬朗的蓝色A30D70环形光源 + Computar 50mm镜头\眼镜板-镜面十字星和圆\[DEV0]-ID011935-BW.bmp';
imo = imread(pname);
figure;imshow(imo);

OstuT = graythresh(imo);
imbw = im2bw(imo,OstuT);
imbw = ~imbw;
figure;imshow(imbw);

se = strel('disk',10);
imc = imclose(imbw,se);
figure;imshow(imc);

% imskel=bwmorph(imc,'skel',Inf);
% figure;imshow(imskel);
figure;imshow(imc)
hold on

[H,theta,rho]=hough(imc,'ThetaResolution',0.5,'RhoResolution',0.2);
P=houghpeaks(H,100,'threshold',ceil(0.2*max(H(:))));
lines=houghlines(imc,theta,rho,P,'FillGap',10,'MinLength',200);
for k=1:length(lines)
    xy=[lines(k).point1;lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',3,'Color','red');
end


imd = abs(double(imbw)-double(imc));
figure;imshow(imd)


% h = fspecial('gaussian',3,0.5);
% imsmooth = imfilter(imo,h);
% figure;imshow(imsmooth);
% 
% imedge=edge(imsmooth,'canny',[0.05 0.12],3.0);
% figure;imshow(imedge);