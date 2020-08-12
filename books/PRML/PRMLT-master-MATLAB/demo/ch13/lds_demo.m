% demos for LDS in ch13

clear; close all;
d = 3;
k = 2;
n = 100;
 
[X,Z,model] = ldsRnd(d,k,n);
[mu, V, llh] = kalmanFilter(model, X);

[nu, U, Ezz, Ezy, llh] = kalmanSmoother(model, X);
% [model, llh] = ldsEm(X,k);
% plot(llh);
% 
