% demos for ch04

%% Logistic logistic regression for binary classification
close all;
clear; 
d = 2;
k = 2;
n = 1000;
[X,y] = kmeansRnd(d,k,n);
[model, llh] = logitBin(X,y-1);
plot(llh);
t = logitBinPred(model,X)+1;
figure
binPlot(model,X,y)