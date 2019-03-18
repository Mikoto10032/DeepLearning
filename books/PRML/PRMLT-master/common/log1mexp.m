function y = log1mexp(x)
% Accurately compute y = log(1-exp(x))
% reference: Accurately Computing log(1-exp(-|a|)) Martin Machler
y = x;
i = x < -log(2);
y(i) = log1p(-exp(x(i)));
y(~i) = log(-expm1(x(~i)));
