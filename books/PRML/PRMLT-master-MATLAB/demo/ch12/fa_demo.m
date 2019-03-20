% demos for ch12

clear; close all;
d = 3;
m = 2;
n = 1000;

X = ppcaRnd(m,d,n);
plotClass(X);

%% Factor analysis
[W, mu, psi, llh] = fa(X, m);
plot(llh);
