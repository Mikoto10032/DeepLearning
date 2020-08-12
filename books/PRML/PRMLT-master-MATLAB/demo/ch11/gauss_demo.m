
%% Sequential update for Gaussian 
close all; clear;
d = 2;
n = 100;
X = randn(d,n);
x = randn(d,1);

mu = mean(X,2);
Xo = bsxfun(@minus,X,mu);
Sigma = Xo*Xo'/n;
p1 = logGauss(x,mu,Sigma);

gauss = Gauss(X(:,3:end)).addSample(X(:,1)).addSample(X(:,2)).addSample(X(:,3)).delSample(X(:,3));
p2 = gauss.logPdf(x);
maxdiff(p1,p2)
%% Sequential update for Gaussian-Wishart
close all; clear;
d = 2;
n = 100;
X = randn(d,n);
x = randn(d,1);

kappa0 = 1;
m0 = zeros(d,1);
nu0 = d;
S0 = eye(d);

xbar = mean(X,2);
kappa = kappa0+n;
nu = nu0+n;
m = (n*xbar+kappa0*m0)/kappa;
Xo = bsxfun(@minus,X,m);
X0 = m0-m;
S = S0+Xo*Xo'+kappa0*(X0*X0');

v = (nu-d+1);
r = (1+1/kappa)/v;
p1 = logSt(x,m,r*S,v);

gw0 = GaussWishart(kappa0,m0,nu0,S0);
gw0 = gw0.addData(X);
p0 = gw0.logPredPdf(x);

gw = GaussWishart(kappa0,m0,nu0,S0);
for i=1:n
    gw = gw.addSample(X(:,i));
end
p2 = gw.logPredPdf(x);
maxdiff(p1,p2)
% 
