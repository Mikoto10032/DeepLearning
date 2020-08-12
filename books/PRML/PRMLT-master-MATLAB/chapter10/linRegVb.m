function [model, energy] = linRegVb(X, t, prior)
% Variational Bayesian inference for linear regression.
% Input:
%   X: d x n data
%   t: 1 x n response
%   prior: prior parameter
% Output:
%   model: trained model structure
%   energy: variational lower bound
% Written by Mo Chen (sth4nth@gmail.com).
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
[m,n] = size(X);
I = eye(m);
xbar = mean(X,2);
tbar = mean(t,2);

X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

XX = X*X';
Xt = X*t';

maxiter = 100;
energy = -inf(1,maxiter+1);
tol = 1e-8;

a = a0+m/2;              % 10.94
c = c0+n/2;
Ealpha = 1e-4;
Ebeta = 1e-4;
for iter = 2:maxiter
%     q(w)
    invS = diag(Ealpha)+Ebeta*XX;            % 10.101
    U = chol(invS);
    Ew = Ebeta*(U\(U'\Xt));                  % 10.100
    KLw = -sum(log(diag(U)));        
%     q(alpha)
    w2 = dot(Ew,Ew);
    invU = U'\I;   
    trS = dot(invU(:),invU(:));
    b = b0+0.5*(w2+trS);                      % 10.95
    Ealpha = a/b;                              % 10.102
    KLalpha = -a*log(b);
%     q(beta)
    e2 = sum((t-Ew'*X).^2);    
    invUX = U'\X;
    trXSX = dot(invUX(:),invUX(:));
    d = d0+0.5*(e2+trXSX);
    Ebeta = c/d; 
    KLbeta = -c*log(d);
%     lower bound
    energy(iter) = KLalpha+KLbeta+KLw;
    if energy(iter)-energy(iter-1) < tol*abs(energy(iter-1)); break; end
end
const = gammaln(a)-gammaln(a0)+gammaln(c)-gammaln(c0)+a0*log(b0)+c0*log(d0)+0.5*(m-n*log(2*pi));
energy = energy(2:iter)+const;
w0 = tbar-dot(Ew,xbar);

model.w0 = w0;
model.w = Ew;
model.alpha = Ealpha;
model.beta = Ebeta;
model.a = a;
model.b = b;
model.c = c;
model.d = d;
model.xbar = xbar;
