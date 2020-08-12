%% Bernoulli Mixture via EM
close all; clear;
d = 2;
k = 3;
n = 5000;
[X,z,mu] = mixBernRnd(d,k,n);
[label,model,llh] = mixBernEm(X,k);
plot(llh);
