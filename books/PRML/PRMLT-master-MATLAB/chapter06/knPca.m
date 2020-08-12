function model = knPca(X, q, kn)
% Kernel PCA
% Input:
%   X: d x n data matrix 
%   q: target dimension
%   kn: kernel function
% Ouput:
%   model: trained model structure
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    kn = @knGauss;
end
K = knCenter(kn,X);
[V,L] = eig(K);
[L,idx] = sort(diag(L),'descend');
V = V(:,idx(1:q));
L = L(1:q);

model.kn = kn;
model.V = V;
model.L = L;
model.X = X;