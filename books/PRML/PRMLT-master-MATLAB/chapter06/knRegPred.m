function [y, sigma, p] = knRegPred(model, Xt, t)
% Prediction for Gaussian Process (kernel) regression model
% Input:
%   model: trained model structure
%   Xt: d x n testing data
%   t (optional): 1 x n testing response
% Output:
%   y: 1 x n prediction
%   sigma: variance
%   p: 1 x n likelihood of t
% Written by Mo Chen (sth4nth@gmail.com).
kn = model.kn;
a = model.a;
X = model.X;
tbar = model.tbar;
Kt = knCenter(kn,X,X,Xt);
y = a'*Kt+tbar;
%% probability prediction 
if nargout > 1
    alpha = model.alpha;
    beta = model.beta;
    U = model.U;
    XU = U'\Kt;
    sigma = sqrt(1/beta+(knCenter(kn,X,Xt)-dot(XU,XU,1))/alpha); 
end

if nargin == 3 && nargout == 3
    p = exp(-0.5*(((t-y)./sigma).^2+log(2*pi))-log(sigma));
end