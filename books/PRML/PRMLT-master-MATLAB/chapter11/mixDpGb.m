function [label, Theta, w, llh] = mixDpGb(X, alpha, theta)
% Collapsed Gibbs sampling for Dirichlet process (infinite) mixture model. 
% Any component model can be used, such as Gaussian.
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
[label,Theta,w] = mixDpGbOl(X,alpha,theta);
nk = n*w;
maxIter = 50;
llh = zeros(1,maxIter);
for iter = 1:maxIter
    for i = randperm(n)
        x = X(:,i);
        k = label(i);
        Theta{k} = Theta{k}.delSample(x);
        nk(k) = nk(k)-1;
        if nk(k) == 0           % remove empty cluster
            Theta(k) = [];
            nk(k) = [];
            which = label>k;
            label(which) = label(which)-1;
        end
        Pk = log(nk)+cellfun(@(t) t.logPredPdf(x), Theta);
        P0 = log(alpha)+theta.logPredPdf(x);
        p = [Pk,P0];
        llh(iter) = llh(iter)+sum(p-log(n));
        k = discreteRnd(exp(p-logsumexp(p)));
        if k == numel(Theta)+1                 % add extra cluster
            Theta{k} = theta.clone().addSample(x);
            nk = [nk,1];
        else
            Theta{k} = Theta{k}.addSample(x);
            nk(k) = nk(k)+1;
        end
        label(i) = k;
    end
end
w = nk/n;

