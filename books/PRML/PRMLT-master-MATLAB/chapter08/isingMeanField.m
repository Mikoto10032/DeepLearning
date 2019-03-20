function mu = isingMeanField(J, h, epoch)
% Mean field for 2d Ising model
% Input: 
%   J: scalar edge potential
%   h: M X N image size node potential
%   edgePot: k x k x m edge potential 
% Output:
%   mu: M x N image size expectation
% Written by Mo Chen (sth4nth@gmail.com)
tol = 0;
if nargin < 3
    epoch = 50;
    tol = 1e-8;
end
[M,N] = size(h);
mu =  tanh(h);
stride = [-1,1,-M,M];
for t = 1:epoch
    mu0 = mu;
    for j = 1:N
        for i = 1:M
            pos = i + M*(j-1);
            ne = pos + stride;
            ne([i,i,j,j] == [1,M,1,N]) = [];
            mu(i,j) = tanh(J*sum(mu(ne)) + h(i,j));
        end
    end
    if max(abs(mu(:)-mu0(:))) < tol; break; end
end 

