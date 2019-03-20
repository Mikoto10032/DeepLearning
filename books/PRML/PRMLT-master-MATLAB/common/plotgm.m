function plotgm(X, model)
% Plot 2d Gaussian mixture model.
% Written by Mo Chen (sth4nth@gmail.com).
level = 64;
n = 256;

spread(X);
x_range = xlim;
y_range = ylim;

x = linspace(x_range(1),x_range(2), n);
y = linspace(y_range(2),y_range(1), n);

[a,b] = meshgrid(x,y);
z = exp(loggmpdf([a(:)';b(:)'],model));

z = z-min(z);
z = floor(z/max(z)*(level-1));

figure;
image(reshape(z,n,n));
colormap(jet(level));
set(gca, 'XTick', [1 256]);
set(gca, 'XTickLabel', [min(x) max(x)]);
set(gca, 'YTick', [1 256]);
set(gca, 'YTickLabel', [min(y) max(y)]);
axis off
