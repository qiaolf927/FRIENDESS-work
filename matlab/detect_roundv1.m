close all
clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%输入

[filename, filepath] = uigetfile( {'*.bmp;*.jpg;*.tif;*.pgm';'*.*'},'打开待识别图片','D:\工作\照片测试\2015-06-26 桌上视觉测试台\');
if filename
    if ~isempty(strfind(filename,'.bmp'))|~isempty(strfind(filename,'.BMP'))|~isempty(strfind(filename,'.jpg'))|~isempty(strfind(filename,'.JPG'))|...
        ~isempty(strfind(filename,'.jpeg'))|~isempty(strfind(filename,'.JPEG'))|~isempty(strfind(filename,'.tif'))|~isempty(strfind(filename,'.TIF'))|...
        ~isempty(strfind(filename,'.tiff'))|~isempty(strfind(filename,'.TIFF'))|~isempty(strfind(filename,'.pgm'))|~isempty(strfind(filename,'.PGM'))
        pname=[filepath,filename];
    end
end
imo = imread(pname);
[ro,co] = size(imo);
figure;imshow(imo);

prompt = {'请输入半径的大小：'};
radiusstr = inputdlg(prompt);
radius = str2num(radiusstr{1});

tic;
%粗定位 降采样处理
scale = 0.5;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imorough = imresize(imo,scale,'bilinear');
radiusrough = round(radius*scale);


OstuT = graythresh(imorough);
imbw = im2bw(imorough,OstuT);
imbw = ~imbw;

imedge=edge(imbw,'canny',[0.05 0.12],3.0);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[r,c] = size(imbw);

abstep = 1;%圆心位置间隔%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thetastep = pi/180;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hough_sum = zeros(ceil(r/abstep),ceil(c/abstep));

for i = 1:r
    for j = 1:c
        if imedge(i,j)
            for theta = thetastep:thetastep:(2*pi)
                xx = radiusrough*cos(theta) + j;
                yy = radiusrough*sin(theta) + i;
                aa = ceil(xx/abstep); %对应 x，即列
                bb = ceil(yy/abstep); %对应 y，即行
                if aa>0 && aa<ceil(c/abstep)+1 && bb>0 && bb<ceil(r/abstep)+1
                    hough_sum(bb,aa) = hough_sum(bb,aa) + 1;
                end 
            end
        end
    end
end

[maxvalue,cc]=max(max(hough_sum));

imresult = zeros(r,c,3);
imresult(:,:,1) = double(imorough);
imresult(:,:,2) = double(imorough);
imresult(:,:,3) = double(imorough);
T = 0.9 * maxvalue%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num = 0;
for i = 1:ceil(r/abstep)
    for j = 1:ceil(c/abstep)
        if hough_sum(i,j) > T
            num = num+1;
            center = [round(i*abstep-abstep/2),round(j*abstep-abstep/2)];%[行，列]
            imresult(max(1,center(1)-1):min(r,center(1)+1),max(1,center(2)-1):min(c,center(2)+1),1) = 255;
            imresult(max(1,center(1)-1):min(r,center(1)+1),max(1,center(2)-1):min(c,center(2)+1),2) = 0;
            imresult(max(1,center(1)-1):min(r,center(1)+1),max(1,center(2)-1):min(c,center(2)+1),3) = 0;
            for theta = thetastep:thetastep:(2*pi)
                xx = round(radiusrough*cos(theta)) + center(2);
                yy = round(radiusrough*sin(theta)) + center(1);
                imresult(max(1,yy-1):min(r,yy+1),max(1,xx-1):min(c,xx+1),1) = 255;
                imresult(max(1,yy-1):min(r,yy+1),max(1,xx-1):min(c,xx+1),2) = 0;
                imresult(max(1,yy-1):min(r,yy+1),max(1,xx-1):min(c,xx+1),3) = 0;
            end
            fprintf(1,'num=%d Center (%d,%d)\n',num,center(1),center(2));
            bbox{num} = [center(1)-radiusrough , center(2)-radiusrough ; center(1)+radiusrough , center(2)+radiusrough];%[左上角点行，列；右下角点行，列]            
        end
    end
end

%回到高分辨率
for i = 1:num
    bbox{i} = round(bbox{i}/scale);
end

%划定区域
for i = 1:num
    if i == 1;
        region{i} = [bbox{1}(1,1)-30 , bbox{1}(1,2)-30 ; bbox{1}(2,1)+30 , bbox{1}(2,2)+30];%可超过图像尺寸
        rcenter{i} = [ (bbox{1}(1,1)+bbox{1}(2,1))/2 , (bbox{1}(1,2) + bbox{1}(2,2))/2 ];
        overlapnum{i} = 1;
    else
        flag = 1;
        for j = 1:length(region)
            [area,ratio] = overlapregion([bbox{i}(1,1)-30 , bbox{i}(1,2)-30 ; bbox{i}(2,1)+30 , bbox{i}(2,2)+30],[(rcenter{j}(1) - radius - 30) , (rcenter{j}(2) - radius - 30) ; (rcenter{j}(1) + radius + 30) , (rcenter{j}(2) + radius + 30)]);
            if ratio >0.7
                rcenter{j} = overlapnum{j}/(overlapnum{j}+1)*rcenter{j} + 1/(overlapnum{j}+1)*[ (bbox{i}(1,1)+bbox{i}(2,1))/2 , (bbox{i}(1,2) + bbox{i}(2,2))/2 ];
                overlapnum{j} = overlapnum{j} + 1;
                region{j} = [(rcenter{j}(1) - radius -10*(min(5,2+overlapnum{j}))) , (rcenter{j}(2) - radius - 10*(min(5,2+overlapnum{j}))) ; (rcenter{j}(1) + radius + 10*(min(5,2+overlapnum{j}))) , (rcenter{j}(2) + radius + 10*(min(5,2+overlapnum{j})))];
                flag = 0;
                break
            end
        end
        if flag
            k = length(region) + 1;
            region{k} = [bbox{i}(1,1)-30 , bbox{i}(1,2)-30 ; bbox{i}(2,1)+30 , bbox{i}(2,2)+30];%可超过图像尺寸
            rcenter{k} = [ (bbox{i}(1,1)+bbox{i}(2,1))/2 , (bbox{i}(1,2) + bbox{i}(2,2))/2 ];
            overlapnum{k} = 1;
        end
    end
end

%精定位，在每一个候选区域region处理
%进行最终的定位与筛选
%正确的区域几乎是目标圆的外接框，将较为中心的像素值化零

OstuTo = graythresh(imo);
imbwo = im2bw(imo,OstuT);
imbwo = ~imbwo;
imedgeo=edge(imbwo,'canny',[0.05 0.12],3.0);
regionjudge = ones(1,length(region));

for i = 1:length(region)    
    imregion{i} = imedgeo(max(1,round(region{i}(1,1))):min(ro,round(region{i}(2,1))),max(1,round(region{i}(1,2))):min(co,round(region{i}(2,2))));
    [region_r,region_c] = size(imregion{i});
    [rnonzero_o,cnonzero_o] = find(imregion{i});
    for k = 1:size(rnonzero_o,1)
        if (rnonzero_o(k,1)-region_r/2)^2 + (cnonzero_o(k,1)-region_c/2)^2 > (radius+30)^2 || (rnonzero_o(k,1)-region_r/2)^2 + (cnonzero_o(k,1)-region_c/2)^2 < (radius-30)^2
            imregion{i}(rnonzero_o(k,1),cnonzero_o(k,1)) = 0;
        end
    end
    [rnonzero,cnonzero] = find(imregion{i});
    %ransac     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    data_num = size(rnonzero,1);
    ransac_n = 30;
    ransac_k = 10;
    ransac_t = 5;
    ransac_d = 0.2*size(rnonzero_o,1);
    
    iteration = 0;
    best_model = [];
    best_consensus_set = [];
    best_error = 20;
    this_error = 20;
    if data_num <ransac_n
        regionjudge(i) = 0;
        continue
    end
    
    while (iteration < ransac_k)
        maybe_inliers = randperm(data_num);
        maybe_inliers = maybe_inliers(1:ransac_n);
        [maybe_model_a,maybe_model_b,maybe_model_R] = lseround(cnonzero(maybe_inliers,1),rnonzero(maybe_inliers,1));
        maybe_model = [maybe_model_a,maybe_model_b,maybe_model_R];
        consensus_set = maybe_inliers;
        for j = setdiff(1:size(rnonzero,1),maybe_inliers)
            if pointerror(maybe_model,[cnonzero(j),rnonzero(j)]) < ransac_t
                consensus_set = [consensus_set,j];
            end
        end
        if length(consensus_set) > ransac_d
            [better_model_a,better_model_b,better_model_R] = lseround(cnonzero(consensus_set,1),rnonzero(consensus_set,1));
            better_model = [better_model_a,better_model_b,better_model_R];
            this_error = modelerrormeasure(better_model,cnonzero(consensus_set,1),rnonzero(consensus_set,1));
            if (this_error < best_error)
                best_model = better_model;
                best_consensus_set = consensus_set;
                best_error = this_error;
            end
        end
        iteration = iteration + 1;
        if this_error < 5
            break
        end
    end
    if isempty(best_model)
        regionjudge(i) = 0;
        continue
    end
    
    a{i} = best_model(1);
    b{i} = best_model(2);
    R{i} = best_model(3);
    
    errorvalue{i} = modelerrormeasure(best_model,cnonzero_o,rnonzero_o);
    iterationvalue{i} = iteration;
    consensus_setnum{i} = length(best_consensus_set);
    if errorvalue{i} > 25 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        regionjudge(i) = 0;
    end
    %[a{i},b{i},R{i}] = lseround(cnonzero,rnonzero);
end

figure;imshow(imo)
hold on

detectnum = 0;
for i = 1:length(region)
    if regionjudge(i)
        detectnum = detectnum + 1;
        centerresult{i} = [(a{i} + max(1,round(region{i}(1,1))) - 1),(b{i} + max(1,round(region{i}(1,2))) - 1)];
        rectangle('Position',[centerresult{i}(2)-R{i},centerresult{i}(1)-R{i},2*R{i},2*R{i}],'Curvature',[1,1],'EdgeColor','r','LineWidth',4);
        plot(centerresult{i}(2),centerresult{i}(1),'b*')
        fprintf(1,'num=%d   Center (%d,%d)   Radius %d \n',detectnum,centerresult{i}(1),centerresult{i}(2),R{i});        
    end
end
toc




