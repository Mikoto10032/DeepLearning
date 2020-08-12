% demos for ch03
clear; close all;
d = 1;
n = 200;
[x,t] = linRnd(d,n);
%% Linear regression
model = linReg(x,t);
[y,sigma] = linRegPred(model,x,t);
plotCurveBar( x, y, sigma );
hold on;
plot(x,t,'o');
hold off;