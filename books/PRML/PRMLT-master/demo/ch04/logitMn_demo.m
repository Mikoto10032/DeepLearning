%% Logistic logistic regression for multiclass classification
clear
k = 3;
n = 1000;
[X,t] = kmeansRnd(2,k,n);
[model, llh] = logitMn(X,t);
y = logitMnPred(model,X);
plotClass(X,y)
