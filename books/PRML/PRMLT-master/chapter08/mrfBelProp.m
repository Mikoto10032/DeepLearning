function [nodeBel, edgeBel] = mrfBelProp(A, nodePot, edgePot, epoch)
% Belief propagation for MRF (Assuming that egdePot is symmetric)
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

[k,n] = size(nodePot);
m = size(edgePot,3);

[s,t,e] = find(tril(A));
A = sparse([s;t],[t;s],[e;e+m]);       % digraph adjacent matrix, where value is message index
mu = ones(k,2*m)/k;                     % message
for iter = 1:epoch
    mu0 = mu;
    for i = 1:n
        in = nonzeros(A(:,i));                      % incoming message index
        nb = nodePot(:,i).*prod(mu(:,in),2);                       % product of incoming message
        for l = in'
            ep = edgePot(:,:,ud(l,m));
            mu(:,rd(l,m)) = normalize(ep*(nb./mu(:,l)));
        end
    end
    if max(abs(mu(:)-mu0(:))) < tol; break; end
end

nodeBel = zeros(k,n);
for i = 1:n
    nodeBel(:,i) = nodePot(:,i).*prod(mu(:,nonzeros(A(:,i))),2);
end
nodeBel = normalize(nodeBel,1);

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

function i = rd(i, m)
% reverse direction edge index
i = mod(i+m-1,2*m)+1;

function i = ud(i, m)
% undirected edge index
i = mod(i-1,m)+1;