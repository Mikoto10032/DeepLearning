function [label, model, llh] = mixBernEm(X, k)
% Perform EM algorithm for fitting the Bernoulli mixture model.
% Input: 
%   X: d x n binary (0/1) data matrix 
%   k: number of cluster
% Output:
%   label: 1 x n cluster label
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
%% initialization
fprintf('EM for mixture model: running ... \n');
X = sparse(X);
n = size(X,2);
label = ceil(k*rand(1,n));  % random initialization
R = full(sparse(1:n,label,1));
tol = 1e-8;
maxiter = 500;
llh = -inf(1,maxiter);
for iter = 2:maxiter
    model = maximization(X,R);
    [R, llh(iter)] = expectation(X,model);
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter)); break; end;
end
[~,label(:)] = max(R,[],2);
llh = llh(2:iter);

function [R, llh] = expectation(X, model)
mu = model.mu;
w = model.w;
R = X'*log(mu)+(1-X)'*log(1-mu)+log(w);
T = logsumexp(R,2);
llh = mean(T); % loglikelihood
R = exp(R-T);

function model = maximization(X, R)
nk = sum(R,1);
w = nk/sum(nk);
mu = (X*R)./nk;
model.mu = mu;
model.w = w;