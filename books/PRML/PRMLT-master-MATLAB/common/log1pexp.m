function y = log1pexp(x)
% Accurately compute y = log(1+exp(x))
% reference: Accurately Computing log(1-exp(-|a|)) Martin Machler
y = x;
i = x > 18;
j = i & (x <= 33.3);
y(~i) = log1p(exp(x(~i)));
y(j) = x(j)+exp(-x(j));
