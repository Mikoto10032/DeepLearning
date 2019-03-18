function i = randp(p)
% Sample a integer in [1:k] with given probability p
i = find(rand<cumsum(normalize(p)),1);
