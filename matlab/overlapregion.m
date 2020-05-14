function [area,ratio] = overlapregion(region1,region2)
r1p1 = region1(1,:);
r1p2 = region1(2,:);
r2p1 = region2(1,:);
r2p2 = region2(2,:);

area = 0;
allarea = 0;
for i = r1p1(1):r1p2(1)
    for j = r1p1(2):r1p2(2)
        if i>=r2p1(1) && i<=r2p2(1) && j>=r2p1(2) && j<=r2p2(2)
            area = area + 1;
        end
        allarea = allarea + 1;
    end
end

ratio = area/allarea;