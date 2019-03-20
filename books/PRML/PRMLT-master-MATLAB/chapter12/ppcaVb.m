function [model, energy] = ppcaVb(X, q, prior)
% Perform variatioanl Bayeisan inference for probabilistic PCA model. 
% Input:
%   X: d x n data matrix
%   q: dimension of target space
% Output:
%   model: trained model structure
%   ernergy: variantional lower bound
% Reference: 
%   Pattern Recognition and Machine Learning by Christopher M. Bishop 
% Written by Mo Chen (sth4nth@gmail.com).
[m,n] = size(X);
if nargin < 3
    a0 = 1e-4;
    b0 = 1e-4;
    c0 = 1e-4;
    d0 = 1e-4;
else
    a0 = prior.a;
    b0 = prior.b;
    c0 = prior.c;
    d0 = prior.d;
end

if nargin < 2
    q = m-1;
end
tol = 1e-6;
maxIter = 500;
energy = -inf(1,maxIter);

mu = mean(X,2);
Xo = bsxfun(@minus, X, mu);
s = dot(Xo(:),Xo(:));
I = eye(q);
% init parameters
a = a0+m/2;
c = c0+m*n/2;
Ealpha = 1e-4;
Ebeta = 1e-4;
EW = rand(q,m); 
EWo = bsxfun(@minus,EW,mean(EW,2));
EWW = EWo*EWo'/m+EW*EW';
for iter = 2:maxIter  
%     q(z)
    LZ = I+Ebeta*EWW;
    V = inv(chol(LZ));                   % inv(LZ) = V*V';
    EZ = LZ\EW*Xo*Ebeta;
    EZZ = n*(V*V')+EZ*EZ';
    KLZ = n*sum(log(diag(V)));           % KLZ = 0.5*n*log(det(inv(LZ)));
%     q(w)
    LW = diag(Ealpha)+Ebeta*EZZ;
    V = inv(chol(LW));                   % inv(LW) = V*V';  
    EW = LW\EZ*Xo'*Ebeta;
    EWW = m*(V*V')+EW*EW';
    KLW = m*sum(log(diag(V)));           % KLW = 0.5*n*log(det(inv(LW)));
%     q(alpha)
    b = b0+diag(EWW)/2;
    Ealpha = a./b;
    KLalpha = -sum(a*log(b));
%     q(beta)
    WZ = EW'*EZ;
    d = d0+(s-2*dot(Xo(:),WZ(:))+dot(EWW(:),EZZ(:)))/2;
    Ebeta = c/d;
    KLbeta = -c*log(d);
%     q(mu)
%     Emu = Ebeta/(lambda+n*Ebeta)*sum(X-WZ,2);

%     lower bound
    energy(iter) = KLalpha+KLbeta+KLW+KLZ;
    if energy(iter)-energy(iter-1) < tol*abs(energy(iter-1)); break; end  
end
energy = energy(2:iter);

model.Z = EZ;
model.W = EW;
model.apha = Ealpha;
model.beta = Ebeta;
model.a = a;
model.b = b;
model.c = c;
model.d = d;
model.mu = mu;