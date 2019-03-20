clear; close all;
%% Original image
load letterX.mat
img = double(X);
img = sign(img-mean(img(:)));

figure;
subplot(2,3,1);
imagesc(img);
title('Original image');
axis image;
colormap gray;
%% Noisy image
y = img + sigma*randn(size(img)); % noisy signal

subplot(2,3,2);
imagesc(y);
title('Noisy image');
axis image;
colormap gray;
%% Parameters
epoch = 50;
J = 1;   % Ising parameter
sigma = 1; % noise level
%% Mean Field
[A, nodePot, edgePot] = im2mrf(y, J, sigma);
[nodeBel, edgeBel] = mrfMeanField(A, nodePot, edgePot, epoch);
lnZ = gibbsEnergy(nodePot, edgePot, nodeBel, edgeBel);
lnZ0 = betheEnergy(A, nodePot, edgePot, nodeBel, edgeBel);
maxdiff(lnZ, lnZ0)

subplot(2,3,4);
imagesc(reshape(nodeBel(1,:),size(img)));
title('Mean Field');
axis image;
colormap gray;
%% Ising Mean Field 
h = reshape(0.5*diff(nodePot),size(img));
mu = isingMeanField(J, h, epoch);
maxdiff(reshape(mu,1,[]), [1,-1]*nodeBel)

subplot(2,3,3);
imagesc(mu)
title('Ising Mean Field');
axis image;
colormap gray;
%% Belief Propagation
[nodeBel,edgeBel] = mrfBelProp(A, nodePot, edgePot, epoch);
lnZ = betheEnergy(A, nodePot, edgePot, nodeBel, edgeBel);

subplot(2,3,5);
imagesc(reshape(nodeBel(1,:),size(img)));
title('Belief propagation');
axis image;
colormap gray;
%% Expectation Propagation
[nodeBel,edgeBel] = mrfExpProp(A, nodePot, edgePot, epoch);
lnZ0 = betheEnergy(A, nodePot, edgePot, nodeBel, edgeBel);
maxdiff(lnZ, lnZ0)

subplot(2,3,6);
imagesc(reshape(nodeBel(1,:),size(img)));
title('Expectation Propagation');
axis image;
colormap gray;
