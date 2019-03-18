function [y, sigma, p] = linRegPred(model, X, t)
% Compute linear regression model reponse y = w'*X+w0 and likelihood
% Input:
%   model: trained model structure
%   X: d x n testing data
%   t (optional): 1 x n testing response
% Output:
%   y: 1 x n prediction
%   sigma: variance
%   p: 1 x n likelihood of t
% Written by Mo Chen (sth4nth@gmail.com).
w = model.w;
w0 = model.w0;
y = w'*X+w0;
%% probability prediction
if nargout > 1
    beta = model.beta;
    if isfield(model,'U')
        U = model.U;        % 3.54
        Xo = bsxfun(@minus,X,model.xbar);
        XU = U'\Xo;
        sigma = sqrt((1+dot(XU,XU,1))/beta);   % 3.59
    else
        sigma = sqrt(1/beta)*ones(1,size(X,2));
    end
end

if nargin == 3 && nargout == 3
    p = exp(-0.5*(((t-y)./sigma).^2+log(2*pi))-log(sigma));
end

