function [nodeBel, edgeBel] = mrfExpProp(A, nodePot, edgePot, epoch)
% Expectation propagation for MRF (Assuming that egdePot is symmetric)
% Input: 
%   A: n x n adjacent matrix of undirected graph, where value is edge index
%   nodePot: k x n node potential
%   edgePot: k x k x m edge potential
% Output:
%   nodeBel: k x n node belief
%   edgeBel: k x k x m edge belief
% Written by Mo Chen (sth4nth@gmail.com)
tol = 0;
if nargin < 4
    epoch = 50;
    tol = 1e-8;
end

nodePot = exp(-nodePot);  
edgePot = exp(-edgePot);

k = size(nodePot,1);
m = size(edgePot,3);

[s,t,e] = find(tril(A));
mu = ones(k,2*m)/k;         % message
nodeBel = normalize(nodePot,1);
for iter = 1:epoch
    mu0 = mu;
    for l = 1:m
        i = s(l);
        j = t(l);
        eij = e(l);
        eji = eij+m;
        ep = edgePot(:,:,eij);

        nodeBel(:,j) = nodeBel(:,j)./mu(:,eij);
        mu(:,eij) = normalize(ep*(nodeBel(:,i)./mu(:,eji)));
        nodeBel(:,j) = normalize(nodeBel(:,j).*mu(:,eij));
        
        nodeBel(:,i) = nodeBel(:,i)./mu(:,eji);
        mu(:,eji) = normalize(ep*(nodeBel(:,j)./mu(:,eij)));
        nodeBel(:,i) = normalize(nodeBel(:,i).*mu(:,eji));
    end
    if max(abs(mu(:)-mu0(:))) < tol; break; end
end

edgeBel = zeros(k,k,m);
for l = 1:m
    eij = e(l);
    eji = eij+m;
    ep = edgePot(:,:,eij);
    nbt = nodeBel(:,t(l))./mu(:,eij);
    nbs = nodeBel(:,s(l))./mu(:,eji);
    eb = (nbt*nbs').*ep;
    edgeBel(:,:,eij) = eb./sum(eb(:));
end
