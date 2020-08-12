function Y = knPcaPred(model, Xt, opt)
% Prediction for kernel PCA
% Input:
%   model: trained model structure
%   X: d x n testing data
%   t (optional): 1 x n testing response
% Output:
%   Y: prejection result of Xt
% Written by Mo Chen (sth4nth@gmail.com).
kn = model.kn;
V = model.V;
L = model.L;
X = model.X;
Y = bsxfun(@times,V'*knCenter(kn,X,X,Xt),1./sqrt(L));
if nargin == 3 && opt.whiten
    Y = bsxfun(@times,Y,1./sqrt(L));
end

