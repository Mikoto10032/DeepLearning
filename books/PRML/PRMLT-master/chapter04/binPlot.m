function binPlot(model, X, t)
% Plot binary classification result for 2d data
% Input:
%   model: trained model structure
%   X: 2 x n data matrix
%   t: 1 x n label
% Written by Mo Chen (sth4nth@gmail.com).
assert(size(X,1) == 2);
w = model.w;
xi = min(X,[],2);
xa = max(X,[],2);
[x1,x2] = meshgrid(linspace(xi(1),xa(1)), linspace(xi(2),xa(2)));

color = 'brgmcyk';
m = length(color);
figure(gcf);
axis equal
clf;
hold on;
view(2);
for i = 1:max(t)
    idc = t==i;
    scatter(X(1,idc),X(2,idc),36,color(mod(i-1,m)+1));
end
y = w(1)*x1+w(2)*x2+w(3);
contour(x1,x2,y,[-0 0]);
hold off;
