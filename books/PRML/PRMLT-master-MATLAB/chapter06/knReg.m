function model = knReg(X, t, lambda, kn)
% Gaussian process (kernel) regression
% Input:
%   X: d x n data
%   t: 1 x n response
%   lambda: regularization parameter
% Output:
%   model: trained model structure
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 4
    kn = @knGauss;
end
if nargin < 3
    lambda = 1e-2;
end
K = knCenter(kn,X);
tbar = mean(t);
U = chol(K+lambda*eye(size(X,2)));    % 6.62
a = U\(U'\(t(:)-tbar));               % 6.68

model.kn = kn;
model.a = a;
model.X = X;
model.tbar = tbar;
%% for probability prediction
y = a'*K+tbar;
beta = 1/mean((t-y).^2);              % 3.21
alpha = lambda*beta;           % lambda=a/b P.153 3.55
model.alpha = alpha;
model.beta = beta;
model.U = U;