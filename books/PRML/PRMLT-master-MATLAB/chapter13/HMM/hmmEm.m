function [model, llh] = hmmEm(x, init)
% EM algorithm to fit the parameters of HMM model (a.k.a Baum-Welch algorithm)
% Input:
%   x: 1 x n integer vector which is the sequence of observations
%   init: model or k
% Output:s
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
n = size(x,2);
X = sparse(x,1:n,1);
d = size(X,1);
if isstruct(init)   % init with a model
    A = init.A;
    E = init.E;
    s = init.s;
elseif numel(init) == 1  % random init with latent k
    k = init;
    s = normalize(rand(k,1),1);  
    A = normalize(rand(k,k),2);
    E = normalize(rand(k,d),2);
end
tol = 1e-4;
maxIter = 100;
llh = -inf(1,maxIter);
for iter = 2:maxIter
    M = E*X;
%     E-step
    [gamma,alpha,beta,c] = hmmSmoother(M,A,s);
    llh(iter) = mean(log(c));
    if llh(iter)-llh(iter-1) < tol*abs(llh(iter-1)); break; end   % check likelihood for convergence
%     M-step 
    s = gamma(:,1);                                                                             % 13.18
    A = normalize(A.*(alpha(:,1:n-1)*(beta(:,2:n).*M(:,2:n)./c(2:n))'),2);      % 13.19 13.43 13.65
    E = (gamma*X')./sum(gamma,2);                            % 13.23
end
model.s = s;
model.A = A;
model.E = E;
llh = llh(2:iter);

function [gamma, alpha, beta, c] = hmmSmoother(M, A, s)
[K,T] = size(M);
At = A';
c = zeros(1,T);
alpha = zeros(K,T);
[alpha(:,1),c(1)] = normalize(s.*M(:,1),1);
for t = 2:T
    [alpha(:,t),c(t)] = normalize((At*alpha(:,t-1)).*M(:,t),1);  % 13.59
end
beta = ones(K,T);
for t = T-1:-1:1
    beta(:,t) = A*(beta(:,t+1).*M(:,t+1))/c(t+1);   % 13.62
end
gamma = alpha.*beta;                  % 13.64
