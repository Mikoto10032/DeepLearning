% demos for ch12

clear; close all;
d = 3;
m = 2;
n = 1000;

X = ppcaRnd(m,d,n);
plotClass(X);

%% Variational Bayesian probabilistic PCA
[model, energy] = ppcaVb(X);
plot(energy);
