function y = besseliLn(nu,x)
% Compute logarithm of besseli function (modified Bessel function of first kind).
% Written by Mo Chen (mochen80@gmail.com).
% TODO: improve precision using the method in 
% Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.  A. Banerjee, I. S. Dhillon, J. Ghosh, and S. Sra
[v,ierr] = besseli(nu,x);
if any(ierr ~= 0) || any(v == Inf)
    error('ERROR: logbesseli');
end
y = log(v);
