function L = mixGaussEvidence(X, model, prior)
% Variational lower bound of the model evidence (log of marginal likelihood)
% This function implements the method in the book PRML. It is equivalent to the bound inside mixGaussVb function.
% Reference: Pattern Recognition and Machine Learning by Christopher M. Bishop (P.474)
% Written by Mo Chen (sth4nth@gmail.com).
alpha0 = prior.alpha;
kappa0 = prior.kappa;
m0 = prior.m;
v0 = prior.v;
M0 = prior.M;

alpha = model.alpha; % Dirichlet
kappa = model.kappa;   % Gaussian
m = model.m;         % Gasusian
v = model.v;         % Whishart
% M = model.M;         % Whishart: inv(W) = V'*V
U = model.U;
R = model.R;
logR = model.logR;

[d,k] = size(m);
nk = sum(R,1); % 10.51

Elogpi = psi(0,alpha)-psi(0,sum(alpha));
Epz = dot(nk,Elogpi);
Eqz = dot(R(:),logR(:));
logCalpha0 = gammaln(k*alpha0)-k*gammaln(alpha0);
Eppi = logCalpha0+(alpha0-1)*sum(Elogpi);
logCalpha = gammaln(sum(alpha))-sum(gammaln(alpha));
Eqpi = dot(alpha-1,Elogpi)+logCalpha;

U0 = chol(M0);
sqrtR = sqrt(R);
xbar = bsxfun(@times,X*R,1./nk); % 10.52

logW = zeros(1,k);
trSW = zeros(1,k);
trM0W = zeros(1,k);
xbarmWxbarm = zeros(1,k);
mm0Wmm0 = zeros(1,k);
for i = 1:k
    Ui = U(:,:,i);
    logW(i) = -2*sum(log(diag(Ui)));      
    
    Xs = bsxfun(@times,bsxfun(@minus,X,xbar(:,i)),sqrtR(:,i)');
    V = chol(Xs*Xs'/nk(i));
    Q = V/Ui;
    trSW(i) = dot(Q(:),Q(:));  % equivalent to tr(SW)=trace(S/M)
    Q = U0/Ui;
    trM0W(i) = dot(Q(:),Q(:));

    q = Ui'\(xbar(:,i)-m(:,i));
    xbarmWxbarm(i) = dot(q,q);
    q = Ui'\(m(:,i)-m0);
    mm0Wmm0(i) = dot(q,q);
end
ElogLambda = sum(psi(0,bsxfun(@minus,v+1,(1:d)')/2),1)+d*log(2)+logW; % 10.65
Epmu = sum(d*log(kappa0/(2*pi))+ElogLambda-d*kappa0./kappa-kappa0*(v.*mm0Wmm0))/2;
logB0 = v0*sum(log(diag(U0)))-0.5*v0*d*log(2)-logMvGamma(0.5*v0,d);
EpLambda = k*logB0+0.5*(v0-d-1)*sum(ElogLambda)-0.5*dot(v,trM0W);

Eqmu = 0.5*sum(ElogLambda+d*log(kappa/(2*pi)))-0.5*d*k;
logB =  -v.*(logW+d*log(2))/2-logMvGamma(0.5*v,d);
EqLambda = 0.5*sum((v-d-1).*ElogLambda-v*d)+sum(logB);

EpX = 0.5*dot(nk,ElogLambda-d./kappa-v.*trSW-v.*xbarmWxbarm-d*log(2*pi));

L = Epz-Eqz+Eppi-Eqpi+Epmu-Eqmu+EpLambda-EqLambda+EpX;