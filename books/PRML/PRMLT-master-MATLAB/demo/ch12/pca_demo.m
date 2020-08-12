% demos for ch12

clear; close all;
d = 3;
m = 2;
n = 1000;

X = ppcaRnd(m,d,n);
plotClass(X);
%% PCA , EM PCA and Constraint EM PCA produce the same result in the sense of reconstruction mseor
% classical PCA
[U,L,mu,mse1] = pca(X,m);
Y = U'*bsxfun(@minus,X,mu);   % projection
Z1 = bsxfun(@times,Y,1./sqrt(L));  % whiten
figure;
plotClass(Y);
figure;
plotClass(Z1);
mse1
% EM PCA
[W2,Z2,mu,mse2] = pcaEm(X,m);
figure;
plotClass(Z1);
mse2
% Contrained EM PCA
[W3,Z3,mu,mse3] = pcaEmC(X,m);
figure;
plotClass(Z1);
mse3
