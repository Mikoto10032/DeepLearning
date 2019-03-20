function [label, model, L] = mixGaussVb(X, m, prior)
% Variational Bayesian inference for Gaussian mixture.
% Input: 
%   X: d x n data matrix
%   m: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or model structure
% Output:
%   label: 1 x n cluster label
%   model: trained model structure
%   L: variational lower bound
% Reference: Pattern Recognition and Machine Learning by Christopher M. Bishop (P.474)
% Written by Mo Chen (sth4nth@gmail.com).
fprintf('Variational Bayesian Gaussian mixture: running ... \n');
[d,n] = size(X);
if nargin < 3
    prior.alpha = 1;
    prior.kappa = 1;
    prior.m = mean(X,2);
    prior.v = d+1;
    prior.M = eye(d);   % M = inv(W)
end
prior.logW = -2*sum(log(diag(chol(prior.M))));

tol = 1e-8;
maxiter = 2000;
L = -inf(1,maxiter);
model = init(X,m,prior);
for iter = 2:maxiter
    model = expect(X,model);
    model = maximize(X,model,prior);
    L(iter) = bound(X,model,prior);
    if abs(L(iter)-L(iter-1)) < tol*abs(L(iter)); break; end
end
L = L(2:iter);
label = zeros(1,n);
[~,label(:)] = max(model.R,[],2);
[~,~,label(:)] = unique(label);

function model = init(X, m, prior)
n = size(X,2);
if isstruct(m)  % init with a model
    model = m;
elseif numel(m) == 1  % random init k
    k = m;
    label = ceil(k*rand(1,n));
    model.R = full(sparse(1:n,label,1,n,k,n));
elseif all(size(m)==[1,n])  % init with labels
    label = m;
    k = max(label);
    model.R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end
model = maximize(X,model,prior);

% Done
function model = maximize(X, model, prior)
alpha0 = prior.alpha;
kappa0 = prior.kappa;
m0 = prior.m;
v0 = prior.v;
M0 = prior.M;
R = model.R;

nk = sum(R,1); % 10.51
alpha = alpha0+nk; % 10.58
kappa = kappa0+nk; % 10.60
v = v0+nk; % 10.63
m = bsxfun(@plus,kappa0*m0,X*R);
m = bsxfun(@times,m,1./kappa); % 10.61

[d,k] = size(m);
U = zeros(d,d,k); 
logW = zeros(1,k);
r = sqrt(R');
for i = 1:k
    Xm = bsxfun(@minus,X,m(:,i));
    Xm = bsxfun(@times,Xm,r(i,:));
    m0m = m0-m(:,i);
    M = M0+Xm*Xm'+kappa0*(m0m*m0m');     % equivalent to 10.62
    U(:,:,i) = chol(M);
    logW(i) = -2*sum(log(diag(U(:,:,i))));      
end

model.alpha = alpha;
model.kappa = kappa;
model.m = m;
model.v = v;
model.U = U;
model.logW = logW;

% Done
function model = expect(X, model)
alpha = model.alpha; % Dirichlet
kappa = model.kappa;   % Gaussian
m = model.m;         % Gasusian
v = model.v;         % Whishart
U = model.U;         % Whishart 
logW = model.logW;
n = size(X,2);
[d,k] = size(m);

EQ = zeros(n,k);
for i = 1:k
    Q = (U(:,:,i)'\bsxfun(@minus,X,m(:,i)));
    EQ(:,i) = d/kappa(i)+v(i)*dot(Q,Q,1);    % 10.64
end
ElogLambda = sum(psi(0,0.5*bsxfun(@minus,v+1,(1:d)')),1)+d*log(2)+logW; % 10.65
Elogpi = psi(0,alpha)-psi(0,sum(alpha)); % 10.66
logRho = -0.5*bsxfun(@minus,EQ,ElogLambda-d*log(2*pi)); % 10.46
logRho = bsxfun(@plus,logRho,Elogpi);   % 10.46
logR = bsxfun(@minus,logRho,logsumexp(logRho,2)); % 10.49
R = exp(logR);

model.logR = logR;
model.R = R;

% Done
function L = bound(X, model, prior)
alpha0 = prior.alpha;
kappa0 = prior.kappa;
v0 = prior.v;
logW0 = prior.logW;
alpha = model.alpha; 
kappa = model.kappa; 
v = model.v;         
logW = model.logW;
R = model.R;
logR = model.logR;
[d,n] = size(X);
k = size(R,2);

Epz = 0;
Eqz = dot(R(:),logR(:));
logCalpha0 = gammaln(k*alpha0)-k*gammaln(alpha0);
Eppi = logCalpha0;
logCalpha = gammaln(sum(alpha))-sum(gammaln(alpha));
Eqpi = logCalpha;
Epmu = 0.5*d*k*log(kappa0);
Eqmu = 0.5*d*sum(log(kappa));
logB0 = -0.5*v0*(logW0+d*log(2))-logMvGamma(0.5*v0,d);
EpLambda = k*logB0;
logB =  -0.5*v.*(logW+d*log(2))-logMvGamma(0.5*v,d);
EqLambda = sum(logB);
EpX = -0.5*d*n*log(2*pi);
L = Epz-Eqz+Eppi-Eqpi+Epmu-Eqmu+EpLambda-EqLambda+EpX;