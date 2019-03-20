function [nodeBel, edgeBel] = mrfMeanField(A, nodePot, edgePot, epoch)
% Mean field for MRF (Assuming that egdePot is symmetric)
% p(x)=exp(-E(x))/Z, E(x)=\sum(edgePot)+sum(nodePot)
% Input: 
%   A: n x n adjacent matrix of undirected graph, where value is edge index
%   nodePot: k x n node potential 
%   edgePot: k x k x m edge potential 
% Output:
%   nodeBel: k x n node belief q(x_i)
%   edgeBel: k x k x m edge belief q(x_i,x_j)
% Written by Mo Chen (sth4nth@gmail.com)
tol = 0;
if nargin < 4
    epoch = 50;
    tol = 1e-8;
end
[nodeBel,L] = softmax(-nodePot,1);    % init nodeBel    
for iter = 1:epoch
    nodeBel0 = nodeBel;
    for i = 1:numel(L)
        [~,j,e] = find(A(i,:));             % neighbors
        nodeBel(:,i) = softmax(-nodePot(:,i)-reshape(edgePot(:,:,e),2,[])*reshape(nodeBel(:,j),[],1));
    end
    if max(abs(nodeBel(:)-nodeBel0(:))) < tol; break; end
end

[s,t,e] = find(tril(A));
edgeBel = zeros(size(edgePot));
for l = 1:numel(e)
    edgeBel(:,:,e(l)) = nodeBel(:,s(l))*nodeBel(:,t(l))';
end