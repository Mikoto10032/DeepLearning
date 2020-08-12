close all; clear;
d = 2;
k = 3;
n = 5000;
[X,label] = kmeansRnd(d,k,n);
init = ceil(k*rand(1,n));
[y, idx, v] = kmedoids(X,init);
plotClass(X,label);
figure;
plotClass(X,y);

