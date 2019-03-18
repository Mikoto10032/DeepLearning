function r = loggmpdf(X, model)
% Compute log pdf of a Gaussian mixture model.
% Written by Mo Chen (sth4nth@gmail.com).
mu = model.mu;
Sigma = model.Sigma;
w = model.weight;

n = size(X,2);
k = size(mu,2);
logRho = zeros(k,n);

for i = 1:k
    logRho(i,:) = loggausspdf(X,mu(:,i),Sigma(:,:,i));
end
r = logsumexp(bsxfun(@plus,logRho,log(w)'),1);


function y = loggausspdf(X, mu, Sigma)
d = size(X,1);
X = bsxfun(@minus,X,mu);
[U,p]= chol(Sigma);
if p ~= 0
    error('ERROR: Sigma is not PD.');
end
Q = U'\X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;