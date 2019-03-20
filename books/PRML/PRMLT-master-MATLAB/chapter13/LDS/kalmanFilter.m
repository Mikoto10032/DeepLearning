function [mu, V, llh] = kalmanFilter(model, X)
% Kalman filter (forward algorithm for linear dynamic system)
% NOTE: This is the exact implementation of the Kalman filter algorithm in PRML.
% However, this algorithm is not practical. It is numerical unstable. 
% Input:
%   X: d x n data matrix
%   model: model structure
% Output:
%   mu: q x n matrix of latent mean mu_t=E[z_t] w.r.t p(z_t|x_{1:t})
%   V: q x q x n latent covariance U_t=cov[z_t] w.r.t p(z_t|x_{1:t})
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
A = model.A; % transition matrix 
G = model.G; % transition covariance
C = model.C; % emission matrix
S = model.S;  % emision covariance
mu0 = model.mu0; % prior mean
P = model.P0;  % prior covairance

n = size(X,2);
k = size(mu0,1);
mu = zeros(k,n);
V = zeros(k,k,n);
llh = zeros(1,n);
I = eye(k);

PC = P*C';
R = C*PC+S;
K = PC/R;                                        % 13.97
mu(:,1) = mu0+K*(X(:,1)-C*mu0);                     % 13.94
V(:,:,1) = (I-K*C)*P;                               % 13.95
llh(1) = logGauss(X(:,1),C*mu0,R);
for i = 2:n
    [mu(:,i), V(:,:,i), llh(i)] = ...
        forwardStep(X(:,i), mu(:,i-1), V(:,:,i-1), A, G, C, S, I);
end
llh = sum(llh);

function [mu, V, llh] = forwardStep(x, mu, V, A, G, C, S, I)
P = A*V*A'+G;                                               % 13.88
PC = P*C';                                                      
R = C*PC+S;
K = PC/R;                                                   % 13.92
Amu = A*mu;
CAmu = C*Amu;                                               
mu = Amu+K*(x-CAmu);                                        % 13.89
V = (I-K*C)*P;                                              % 13.90
llh = logGauss(x,CAmu,R);                                   % 13.91