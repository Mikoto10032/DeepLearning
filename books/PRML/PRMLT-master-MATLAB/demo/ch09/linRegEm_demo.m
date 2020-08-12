%% Empirical Bayesian linear regression via EM
close all; clear;
d = 5;
n = 200;
[x,t] = linRnd(d,n);
[model,llh] = linRegEm(x,t);
plot(llh);
