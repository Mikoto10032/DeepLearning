function [label, model, energy] = knKmeans(X, init, kn)
% Perform kernel kmeans clustering.
% Input:
%   K: n x n kernel matrix
%   init: either number of clusters (k) or initial label (1xn)
% Output:
%   label: 1 x n sample labels
%   model: trained model structure
%   energy: optimization target value
% Reference: Kernel Methods for Pattern Analysis
% by John Shawe-Taylor, Nello Cristianini
% Written by Mo Chen (sth4nth@gmail.com).
n = size(X,2);
if numel(init)==1
    k = init;
    label = ceil(k*rand(1,n));
elseif numel(init)==n
    label = init;
end
if nargin < 3
    kn = @knGauss;
end
K = kn(X,X);
last = zeros(1,n);
while any(label ~= last)
    [~,~,last(:)] = unique(label);   % remove empty clusters
    E = sparse(last,1:n,1);
    E = E./sum(E,2);
    T = E*K;
    [val, label] = max(T-dot(T,E,2)/2,[],1);
end
energy = trace(K)-2*sum(val); 
if nargout == 3
    model.X = X;
    model.label = label;
    model.kn = kn;
end
