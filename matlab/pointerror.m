function error = pointerror(roundmodel,point)
a = roundmodel(1);%ÐÐ£¬¼´y
b = roundmodel(2);
R = roundmodel(3);
x = point(1);
y = point(2);
error = abs(R-((x-b)^2+(y-a)^2)^(1/2));