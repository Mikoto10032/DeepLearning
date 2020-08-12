function A = lattice( sz )
% Create an undirected graph corresponding to sz lattice
% Example:
%   plot(graph(lattice([2,2,3])))
% Input:
%   sz: 1 x d size of lattice
% Output:
%   A: prod(sz) x prod(sz) adjacent matrix of an undirected graph
% Written by Mo Chen (sth4nth@gmail.com)
d = numel(sz);
step = cumprod(sz);
n = step(end);
M = reshape(1:n,sz);
S = arrayfun(@(i) reshape(slice(M,i,1:sz(i)-1),1,[]), 1:d,'UniformOutput',false);
T = arrayfun(@(i) reshape(slice(M,i,2:sz(i)),1,[]), 1:d,'UniformOutput',false);
A = sparse([S{:}],[T{:}],1,n,n);
A = A+A';