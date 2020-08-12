%% Collapse Gibbs sampling for Dirichelt process gaussian mixture model
close all; clear;
d = 2;
k = 3;
n = 500;
[X,z] = mixGaussRnd(d,k,n);
plotClass(X,z);

[z,Theta,w,llh] = mixGaussGb(X);
figure
plotClass(X,z);

[X,z] = mixGaussSample(Theta,w,n);
figure
plotClass(X,z);

