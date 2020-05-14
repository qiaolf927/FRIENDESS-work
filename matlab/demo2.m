close all
clear
clc

radius = 115;
pname = 'D:\工作\照片\铜板\铜板弱光源曝光800[DEV0]-ID008794-BW.bmp';
imo = imread(pname);
figure;imshow(imo);

OstuT = graythresh(imo);
imbw = im2bw(imo,OstuT);
imbw = ~imbw;
figure;imshow(imbw);

imedge=edge(imbw,'canny',[0.05 0.12],3.0);
figure;imshow(imedge);

imbw = imedge;

[r,c] = size(imbw);
hough_sum = zeros(r+2*radius,c+2*radius);
for i = 1:r
    for j = 1:c
        if imbw(i,j)
            for l =-radius:10:radius
                if i+radius+l>=1+1&&i+radius+l<=r+2*radius-1&&j+radius+round((radius^2-l^2)^(1/2))>=1+1&&j+radius+round((radius^2-l^2)^(1/2))<=c+2*radius-1
                    %右半圆周上点的八邻域内的所有点投票数都加一
                    hough_sum(i+radius+l-1:i+radius+l+1,j+radius+round((radius^2-l^2)^(1/2))-1:j+radius+round((radius^2-l^2)^(1/2))+1)=hough_sum(i+radius+l-1:i+radius+l+1,j+radius+round((radius^2-l^2)^(1/2))-1:j+radius+round((radius^2-l^2)^(1/2))+1)+ones(3);
                end
                if i+radius+l>=1+1&&i+radius+l<=r+2*radius-1&&j+radius-round((radius^2-l^2)^(1/2))>=1+1&&j+radius-round((radius^2-l^2)^(1/2))<=c+2*radius-1
                    %左半圆周上点的八邻域内的所有点投票数都加一
                    hough_sum(i+radius+l-1:i+radius+l+1,j+radius-round((radius^2-l^2)^(1/2))-1:j+radius-round((radius^2-l^2)^(1/2))+1)=hough_sum(i+radius+l-1:i+radius+l+1,j+radius-round((radius^2-l^2)^(1/2))-1:j+radius-round((radius^2-l^2)^(1/2))+1)+ones(3);
                end
            end
        end
    end
end

[maxvalue,cc]=max(max(hough_sum));

imresult = zeros(r,c,3);
imresult(:,:,1) = 255*double(imbw);
imresult(:,:,2) = 255*double(imbw);
imresult(:,:,3) = 255*double(imbw);
T = 0.9 * maxvalue
num = 0;
for i = 1:r+2*radius
    for j = 1:c+2*radius
        if hough_sum(i,j) > T
            num = num+1;
            ii = i - radius;
            jj = j - radius;
            if ii>=1 && ii<=r && jj>=1 && jj<=c
               imresult(ii,jj,1) = 255;
               imresult(ii,jj,2) = 0;
               imresult(ii,jj,3) = 0;
            end
            for l =-radius:radius
                if ii+l>=1&&ii+l<=r&&jj+round((radius^2-l^2)^(1/2))>=1&&jj+round((radius^2-l^2)^(1/2))<=c
                    %右半圆
                    imresult(ii+l,jj+round((radius^2-l^2)^(1/2)),1) = 255;
                    imresult(ii+l,jj+round((radius^2-l^2)^(1/2)),2) = 0;
                    imresult(ii+l,jj+round((radius^2-l^2)^(1/2)),3) = 0;
                end
                if ii+l>=1&&ii+l<=r&&jj-round((radius^2-l^2)^(1/2))>=1&&jj-round((radius^2-l^2)^(1/2))<=c
                    %左半圆
                    imresult(ii+l,jj-round((radius^2-l^2)^(1/2)),1) = 255;
                    imresult(ii+l,jj-round((radius^2-l^2)^(1/2)),2) = 0;
                    imresult(ii+l,jj-round((radius^2-l^2)^(1/2)),3) = 0;
                end
            end
        end
    end
end
figure;imshow(uint8(imresult))



% T = zeros(350);
% [T_r,T_c] = size(T);
% for i = 1:T_r
%     for j = 1:T_c
%         rr = ((i - T_r/2)^2 + (j - T_c/2)^2);
%         if ((radius-5)^2 < rr) && (rr <  (radius+5)^2)
%             T(i,j) = 255;
%         end
%     end
% end
% TT = uint8(T);
% 
% figure;imshow(TT)
% 
% [ro,co] = size(imbw);
% [rt,ct] = size(TT);
% 
% box = phase_corr_match(imbw,TT);
% 
% figure;imshow(imbw)
% hold on
% plot([box(1,1),box(1,2)],[box(2,1),box(2,2)],'r','LineWidth',2);
% plot([box(1,2),box(1,3)],[box(2,2),box(2,3)],'r','LineWidth',2);
% plot([box(1,3),box(1,4)],[box(2,3),box(2,4)],'r','LineWidth',2);
% plot([box(1,4),box(1,1)],[box(2,4),box(2,1)],'r','LineWidth',2);
% hold off