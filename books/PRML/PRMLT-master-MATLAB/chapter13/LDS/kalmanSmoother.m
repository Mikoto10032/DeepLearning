function [nu, U, Ezz, Ezy, llh] = kalmanSmoother(model, X)
% Kalman smoother (forward-backward algorithm for linear dynamic system)
% NOTE: This is the exact implementation of the Kalman smoother algorithm in PRML.
% However, this algorithm is not practical. It is numerical unstable. 
% Input:
%   X: d x n data matrix
%   model: model structure
% Output:
%   nu: q x n matrix of latent mean mu_t=E[z_t] w.r.t p(z_t|x_{1:T})
%   U: q x q x n latent covariance U_t=cov[z_t] w.r.t p(z_t|x_{1:T})
%   Ezz: q x q matrix E[z_tz_t^T]
%   Ezy: q x q matrix E[z_tz_{t-1}^T]
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
A = model.A; % transition matrix 
G = model.G; % transition covariance
C = model.C; % emission matrix
S = model.S;  % emision covariance
mu0 = model.mu0; % prior mean
P0 = model.P0;  % prior covairance

n = size(X,2);
q = size(mu0,1);
mu = zeros(q,n);
V = zeros(q,q,n);
P = zeros(q,q,n); % C_{t+1|t}
Amu = zeros(q,n); % u_{t+1|t}
llh = zeros(1,n);
I = eye(q);

% forward
PC = P0*C';
R = C*PC+S;
K = PC/R;
mu(:,1) = mu0+K*(X(:,1)-C*mu0);
V(:,:,1) = (I-K*C)*P0;
P(:,:,1) = P0;  % useless, just make a point
Amu(:,1) = mu0; % useless, just make a point
llh(1) = logGauss(X(:,1),C*mu0,R);
for i = 2:n    
    [mu(:,i), V(:,:,i), Amu(:,i), P(:,:,i), llh(i)] = ...
        forwardStep(X(:,i), mu(:,i-1), V(:,:,i-1), A, G, C, S, I);
end
llh = sum(llh);
% backward
nu = zeros(q,n);
U = zeros(q,q,n);
Ezz = zeros(q,q,n);
Ezy = zeros(q,q,n-1);

nu(:,n) = mu(:,n);
U(:,:,n) = V(:,:,n);
Ezz(:,:,n) = U(:,:,n)+nu(:,n)*nu(:,n)';
for i = n-1:-1:1  
    [nu(:,i), U(:,:,i), Ezz(:,:,i), Ezy(:,:,i)] = ...
        backwardStep(nu(:,i+1), U(:,:,i+1), mu(:,i), V(:,:,i), Amu(:,i+1), P(:,:,i+1), A);
end

function [mu, V, Amu, P, llh] = forwardStep(x, mu0, V0, A, G, C, S, I)
P = A*V0*A'+G;                                              % 13.88
PC = P*C';
R = C*PC+S;
K = PC/R;                                                   % 13.92
Amu = A*mu0;
CAmu = C*Amu;
mu = Amu+K*(x-CAmu);                                        % 13.89
V = (I-K*C)*P;                                              % 13.90
llh = logGauss(x,CAmu,R);                                   % 13.91


function [nu, U, Ezz, Ezy] = backwardStep(nu0, U0, mu, V, Amu, P, A)
J = V*A'/P;                                                 % 13.102
nu = mu+J*(nu0-Amu);                                        % 13.100
U = V+J*(U0-P)*J';                                          % 13.101
Ezy = J*U0+nu0*nu';                                         % 13.106 
Ezz = U+nu*nu';                                             % 13.107