function y = sigmoid(x)
% Sigmod function
% Written by Mo Chen (sth4nth@gmail.com).
y = exp(-log1pexp(-x));