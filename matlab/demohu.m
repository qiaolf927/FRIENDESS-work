close all
clear
clc

radius = 150;
pname = 'D:\����\��Ƭ����\2015-06-18 γ�ʵ���ɫA30D70���ι�Դ + Computar 50mm��ͷ\�۾���-����ʮ���Ǻ�Բ\[DEV0]-ID015452-BW.bmp';
imo = imread(pname);
figure;imshow(imo);

OstuT = graythresh(imo);
imbw = im2bw(imo,OstuT);
imbw = ~imbw;
figure;imshow(imbw);

[r,c] = size(imbw);

T = zeros(350);
[T_r,T_c] = size(T);
for i = 1:T_r
    for j = 1:T_c
        rr = ((i - T_r/2)^2 + (j - T_c/2)^2);
        if ((radius-5)^2 < rr) && (rr <  (radius+5)^2)
            T(i,j) = 1;
        end
    end
end
Hu_T = Hufeature(T);
diffmin = -1;

for i = 1:10:(r-T_r+1)
    for j = 1:10:(c-T_c+1)
        impart = imbw(i:i+T_r-1,j:j+T_c-1);
        Hu_part = Hufeature(impart);
        diff = (Hu_T-Hu_part)*(Hu_T-Hu_part)';
        if diffmin < 0
            diffmin = diff;
            istart = i;
            jstart = j;
        end
        if diff < diffmin
            diffmin = diff;
            istart = i;
            jstart = j;
        end
    end
end

figure;imshow(imbw)
hold on
plot([j,j],[i,i+T_r],'r','LineWidth',2);
plot([j,j+T_c],[i+T_r,i+T_r],'r','LineWidth',2);
plot([j+T_c,j+T_c],[i+T_r,i],'r','LineWidth',2);
plot([j+T_c,j],[i,i],'r','LineWidth',2);