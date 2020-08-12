function [label, Theta, w, llh] = mixDpGbOl(X, alpha, theta)
% Online collapsed Gibbs sampling for Dirichlet process (infinite) mixture model. 
% Input: 
%   X: d x n data matrix
%   alpha: parameter for Dirichlet process prior
%   theta: class object for prior of component distribution (such as Gauss)
% Output:
%   label: 1 x n cluster label
%   Theta: 1 x k structure of trained components
%   w: 1 x k component weight vector
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
Theta = {};
nk = [];
label = zeros(1,n);
llh = 0;
for i = randperm(n)
    x = X(:,i);
    Pk = log(nk)+cellfun(@(t) t.logPredPdf(x), Theta);
    P0 = log(alpha)+theta.logPredPdf(x);
    p = [Pk,P0];
    llh = llh+sum(p-log(n));
    k = discreteRnd(exp(p-logsumexp(p)));
    if k == numel(Theta)+1
        Theta{k} = theta.clone().addSample(x);
        nk = [nk,1];
    else
        Theta{k} = Theta{k}.addSample(x);
        nk(k) = nk(k)+1;
    end
    label(i) = k;
end
w = nk/n;