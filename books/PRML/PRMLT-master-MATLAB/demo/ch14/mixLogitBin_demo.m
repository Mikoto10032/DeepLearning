%% Mixture of logistic regression
d = 2;
c = 4;
k = 4;
n = 500;
[X,t] = kmeansRnd(d,c,n);

model = mixLogitBin(X,t-1,k);
y = mixLogitBinPred(model,X);
plotClass(X,y+1)