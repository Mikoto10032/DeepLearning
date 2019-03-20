
% demos for ch01
clear;
k = 10;  % variable range
n = 100;  % number of variables

x = ceil(k*rand(1,n));
y = ceil(k*rand(1,n));

%% Entropy H(x), H(y)
Hx = entropy(x);
Hy = entropy(y);
%% Joint entropy H(x,y)
Hxy = jointEntropy(x,y);
%% Conditional entropy H(x|y)
Hx_y = condEntropy(x,y);
%% Mutual information I(x,y)
Ixy = mutInfo(x,y);
%% Relative entropy (KL divergence) KL(p(x)|p(y))
Dxy = relatEntropy(x,y);
%% Normalized mutual information I_n(x,y)
nIxy = nmi(x,y);
%% Nomalized variation information I_v(x,y)
vIxy = nvi(x,y);
%% H(x|y) = H(x,y)-H(y)
isequalf(Hx_y,Hxy-Hy)
%% I(x,y) = H(x)-H(x|y)
isequalf(Ixy,Hx-Hx_y)
%% I(x,y) = H(x)+H(y)-H(x,y)
isequalf(Ixy,Hx+Hy-Hxy)
%% I_n(x,y) = I(x,y)/sqrt(H(x)*H(y))
isequalf(nIxy,Ixy/sqrt(Hx*Hy))
%% I_v(x,y) = (1-I(x,y)/H(x,y))
isequalf(vIxy,1-Ixy/Hxy)



