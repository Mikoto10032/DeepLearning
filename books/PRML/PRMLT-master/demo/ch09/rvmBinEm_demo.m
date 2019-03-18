%% RVM classification via EM
clear; close all
k = 2;
d = 2;
n = 1000;
[X,t] = kmeansRnd(d,k,n);
[x1,x2] = meshgrid(linspace(min(X(1,:)),max(X(1,:)),n), linspace(min(X(2,:)),max(X(2,:)),n));

[model, llh] = rvmBinEm(X,t-1);
plot(llh);
y = rvmBinPred(model,X)+1;
figure;
binPlot(model,X,y);