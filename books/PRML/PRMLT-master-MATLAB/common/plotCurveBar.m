function plotCurveBar( x, y, sigma )
% Plot 1d curve and variance
% Input:
%   x: 1 x n 
%   y: 1 x n 
%   sigma: 1 x n or scaler 
% Written by Mo Chen (sth4nth@gmail.com).
color = [255,228,225]/255; %pink
[x,idx] = sort(x);
y = y(idx);
sigma = sigma(idx);

fill([x,fliplr(x)],[y+sigma,fliplr(y-sigma)],color);
hold on;
plot(x,y,'r-');
hold off
axis([x(1),x(end),-inf,inf])

