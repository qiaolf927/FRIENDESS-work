function error = modelerrormeasure(model,X,Y)
num = size(X,1);
error = 0;
for i = 1:num
    error = error + pointerror(model,[X(i,1),Y(i,1)]);
end
error = error/num;