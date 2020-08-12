function [model, llh] = rvmRegSeq(X, t)
% TODO: beta is not updated.
% Sparse Bayesian Regression (RVM) using sequential algorithm
% Input:
%   X: d x n data
%   t: 1 x n response
% Output:
%   model: trained model structure
%   llh: loglikelihood 
% reference:
% Tipping and Faul. Fast marginal likelihood maximisation for sparse Bayesian models. AISTATS 2003.
% Written by Mo Chen (sth4nth@gmail.com).
maxiter = 1000;
llh = -inf(1,maxiter);
tol = 1e-4;

[d,n] = size(X);
xbar = mean(X,2);
tbar = mean(t,2);
X = bsxfun(@minus,X,xbar);
t = bsxfun(@minus,t,tbar);

beta = 1/mean(t.^2);   % beta = 1/sigma^2
alpha = inf(d,1);
S = beta*dot(X,X,2);     % eq.(22)
Q = beta*(X*t');         % eq.(22)
Sigma = zeros(0,0);  
mu = zeros(0,1);  
index = zeros(0,1);
Phi = zeros(0,n);
iAct = zeros(d,3);   
for iter = 2:maxiter
    s = S; q = Q; % p.353 Execrcies 7.17
    s(index) = alpha(index).*S(index)./(alpha(index)-S(index)); % 7.104
    q(index) = alpha(index).*Q(index)./(alpha(index)-S(index)); % 7.105    

    theta = q.^2-s;
    iNew = theta>0;

    iUse = false(d,1);
    iUse(index) = true;
    
    iUpd = (iNew & iUse); % update
    iAdd = (iNew ~= iUpd); % add
    iDel = (iUse ~= iUpd); % del
    
    dllh = -inf(d,1);  % delta likelihood (likelihood improvement of each step, eventually approches 0.)
    if any(iUpd)
        alpha_ = s(iUpd).^2./theta(iUpd);      % eq.(20)
        delta = 1./alpha_-1./alpha(iUpd);
        dllh(iUpd) = Q(iUpd).^2.*delta./(S(iUpd).*delta+1)-log1p(S(iUpd).*delta);  % eq.(32)
    end
    if any(iAdd)
        dllh(iAdd) = (Q(iAdd).^2-S(iAdd))./S(iAdd)+log(S(iAdd)./(Q(iAdd).^2));    % eq.(27)
    end
    if any(iDel)
        dllh(iDel) = Q(iDel).^2./(S(iDel)-alpha(iDel))-log1p(-S(iDel)./alpha(iDel));  % eq.(37)
    end

    [llh(iter),j] = max(dllh);
    if llh(iter) < tol; break; end

    iAct(:,1) = iUpd;
    iAct(:,2) = iAdd;
    iAct(:,3) = iDel;
    
    % update parameters
    switch find(iAct(j,:))
        case 1 % update: 
            idx = (index==j);
            alpha_ = s(j)^2/theta(j);

            Sigma_j = Sigma(:,idx);
            Sigma_jj = Sigma(idx,idx);
            mu_j = mu(idx);
            
            kappa = 1/(Sigma_jj+1/(alpha_-alpha(j)));
            Sigma = Sigma-kappa*(Sigma_j*Sigma_j');                    % eq.(33)
            mu = mu-kappa*mu_j*Sigma_j;                                  % eq.(34)

            v = beta*X*(Phi'*Sigma_j);
            S = S+kappa*v.^2;                   % eq.(35)
            Q = Q+kappa*mu_j*v;                 % eq.(36)
            alpha(j) = alpha_;
        case 2 % Add
            alpha_ = s(j)^2/theta(j);
            Sigma_jj = 1/(alpha_+S(j));
            mu_j = Sigma_jj*Q(j);
            phi_j = X(j,:);             

            v = beta*Sigma*(Phi*phi_j');  
            off = -Sigma_jj*v;                         % eq.(28) has error?
            Sigma = [Sigma+Sigma_jj*(v*v'), off; off', Sigma_jj];   % eq.(28)
            mu = [mu-mu_j*v; mu_j];                                % eq.(29)
            
            e_j = phi_j-v'*Phi;
            v = beta*X*e_j';
            S = S-Sigma_jj*v.^2;                      % eq.(30)
            Q = Q-mu_j*v;                              % eq.(31)
            
            index = [index;j]; %#ok<AGROW>
            alpha(j) = alpha_;
        case 3 % del
            idx = (index==j);
            Sigma_j = Sigma(:,idx);
            Sigma_jj = Sigma(idx,idx);
            mu_j = mu(idx);
            
            Sigma = Sigma-(Sigma_j*Sigma_j')/Sigma_jj;                    % eq.(38)
            mu = mu-mu_j*Sigma_j/Sigma_jj;                                  % eq.(39)

            v = beta*X*(Phi'*Sigma_j);
            S = S+v.^2/Sigma_jj;                   % eq.(40)
            Q = Q+mu_j*v/Sigma_jj;                  % eq.(41)
            
            mu(idx) = [];
            Sigma(:,idx) = [];
            Sigma(idx,:) = [];
            index(idx) = [];
            alpha(j) = inf;
    end
    Phi = X(index,:); 
%     beta = ;
end
llh = cumsum(llh(2:iter));
w0 = tbar-dot(mu,xbar(index));

model.index = index;
model.w0 = w0;
model.w = mu;
model.alpha = alpha(index);
model.beta = beta;