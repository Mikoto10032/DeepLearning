function Kc = knCenter(kn, X, X1, X2)
% Centerize the data in the kernel space
% Input:
%   kn: kernel function
%   X: d x n data matrix of which the center in the kernel space is computed
%   X1, X2: d x n1 and d x n2 data matrix. the kernel k(x1,x2) is computed
%       where the origin of the kernel space is the center of phi(X)
% Ouput:
%   Kc: n1 x n2 kernel matrix between X1 and X2 in kernel space centered by
%       center of phi(X)
% Written by Mo Chen (sth4nth@gmail.com).
K = kn(X,X);
mK = mean(K);
mmK = mean(mK);
if nargin == 2     % compute the pairwise centerized version of the kernel of X. eq knCenter(kn,X,X,X)
    Kc = K+mmK-bsxfun(@plus,mK',mK);        % Kc = K-M*K-K*M+M*K*M; where M = ones(n,n)/n; 
elseif nargin == 3  % compute the norms (k(x,x)) of X1 w.r.t. the center of X as the origin. eq diag(knCenter(kn,X,X1,X1))
    Kc = kn(X1)+mmK-2*mean(kn(X,X1));
elseif nargin == 4  % compute the kernel of X1 and X2 w.r.t. the center of X as the origin
    Kc = kn(X1,X2)+mmK-bsxfun(@plus,mean(kn(X,X1))',mean(kn(X,X2)));
end
