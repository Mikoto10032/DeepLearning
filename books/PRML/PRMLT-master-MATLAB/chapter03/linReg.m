function model = linReg(X, t, lambda)
% Fit linear regression model y=w'x+w0  
% Input:
%   X: d x n data
%   t: 1 x n response
%   lambda: regularization parameter
% Output:
%   model: trained model structure
% Written by Mo Chen (sth4nth@gmail.com).
if nargin < 3
    lambda = 0;
end
d = size(X,1);
idx = (1:d)';
dg = sub2ind([d,d],idx,idx);

xbar = mean(X,2);
tbar = mean(t,2);
X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

XX = X*X';
XX(dg) = XX(dg)+lambda;     % 3.54 XX=inv(S)/beta
% w = XX\(X*t');
U = chol(XX);
w = U\(U'\(X*t'));  % 3.15 & 3.28
w0 = tbar-dot(w,xbar);  % 3.19

model.w = w;
model.w0 = w0;
model.xbar = xbar;
%% for probability prediction
beta = 1/mean((t-w'*X).^2); % 3.21
% alpha = lambda*beta;           % lambda=a/b P.153 3.55
% model.alpha = alpha;
model.beta = beta;
model.U = U;
