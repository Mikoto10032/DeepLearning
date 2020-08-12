%% Naive Bayes with independent Bernoulli
close all; clear;
d = 10;
k = 2;
n = 2000;
[X,t,mu] = mixBernRnd(d,k,n);
m = floor(n/2);
X1 = X(:,1:m);
X2 = X(:,(m+1):end);
t1 = t(1:m);
t2 = t((m+1):end);
model = nbBern(X1,t1);
y2 = nbBernPred(model,X2);
err = sum(t2~=y2)/numel(t2);