
%% adaboost
d = 2;
k = 2;
n = 500;
[X,t] = kmeansRnd(d,k,n);
model = adaboostBin(X,t-1);
y = adaboostBinPred(model,X);
plotClass(X,y+1)

