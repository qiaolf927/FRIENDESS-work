function [a,b] = centeropt(rnonzero,cnonzero,im,radius)
indexleft = 1:size(rnonzero,1);
[r,c] = size(im);
indexnew = [];
asum = 0;
bsum = 0;
for i = indexleft
    temp = (rnonzero(i,1)-r/2)^2 + (cnonzero(i,1)-c/2)^2 ;
    if temp > 0.8*radius^2 && temp < 1.2*radius^2
        indexnew = [indexnew,i];
        asum = asum + rnonzero(i);
        bsum = bsum + cnonzero(i);
    end
end
indexleft = indexnew;
a = asum/length(indexleft);
b = bsum/length(indexleft);

% anew = a;
% bnew = b;
% a = 0;
% b = 0;
% iter = 0;
% while (a-anew)^2 + (b-bnew)^2 > 2 && iter < 10;
%     a = anew;
%     b = bnew;
%     iter = iter + 1;
%     indexnew = [];
%     asum = 0;
%     bsum = 0;
%     for i = indexleft
%         temp = (rnonzero(i,1)-a)^2 + (cnonzero(i,1)-b)^2 ;
%         if temp > 0.8*radius^2 && temp < 1.2*radius^2
%             indexnew = [indexnew,i];
%             asum = asum + rnonzero(i);
%             bsum = bsum + cnonzero(i);
%         end        
%     end
%     indexleft = indexnew;
%     anew = asum/length(indexleft);
%     bnew = bsum/length(indexleft);
%     disp('............................')
%     iter
%     anew
%     bnew
% end
% a = anew;
% b = bnew;
%     
%                             