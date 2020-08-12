function [label, model, llh] = mixGaussEm(X, init)
% Perform EM algorithm for fitting the Gaussian mixture model.
% Input: 
%   X: d x n data matrix
%   init: k (1 x 1) number of components or label (1 x n, 1<=label(i)<=k) or model structure
% Output:
%   label: 1 x n cluster label
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
%% init
fprintf('EM for Gaussian mixture: running ... \n');
tol = 1e-6;
maxiter = 500;
llh = -inf(1,maxiter);
R = initialization(X,init);
for iter = 2:maxiter
    [~,label(1,:)] = max(R,[],2);
    R = R(:,unique(label));   % remove empty clusters
    model = maximization(X,R);
    [R, llh(iter)] = expectation(X,model);
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter)); break; end;
end
llh = llh(2:iter);

function R = initialization(X, init)
n = size(X,2);
if isstruct(init)  % init with a model
    R  = expectation(X,init);
elseif numel(init) == 1  % random init k
    k = init;
    label = ceil(k*rand(1,n));
    R = full(sparse(1:n,label,1,n,k,n));
elseif all(size(init)==[1,n])  % init with labels
    label = init;
    k = max(label);
    R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end

function [R, llh] = expectation(X, model)
mu = model.mu;
Sigma = model.Sigma;
w = model.w;

n = size(X,2);
k = size(mu,2);
R = zeros(n,k);
for i = 1:k
    R(:,i) = loggausspdf(X,mu(:,i),Sigma(:,:,i));
end
R = bsxfun(@plus,R,log(w));
T = logsumexp(R,2);
llh = sum(T)/n; % loglikelihood
R = exp(bsxfun(@minus,R,T));

function model = maximization(X, R)
[d,n] = size(X);
k = size(R,2);
nk = sum(R,1);
w = nk/n;
mu = bsxfun(@times, X*R, 1./nk);

Sigma = zeros(d,d,k);
r = sqrt(R);
for i = 1:k
    Xo = bsxfun(@minus,X,mu(:,i));
    Xo = bsxfun(@times,Xo,r(:,i)');
    Sigma(:,:,i) = Xo*Xo'/nk(i)+eye(d)*(1e-6);
end

model.mu = mu;
model.Sigma = Sigma;
model.w = w;

function y = loggausspdf(X, mu, Sigma)
d = size(X,1);
X = bsxfun(@minus,X,mu);
[U,p]= chol(Sigma);
if p ~= 0
    error('ERROR: Sigma is not PD.');
end
Q = U'\X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;