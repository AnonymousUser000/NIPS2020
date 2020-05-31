function [pr] = pagerank_4Large(pmatrix, vecD)
[n, n] = size(pmatrix);

x = ones(1, n) / n;
z = ones(1, n);
maxIter = 100;
numIter = 0;
while max(abs(x - z) > 0.0001)
    z = x;
    x = x * pmatrix;
    xSum = sum(x);
    x = x + ones(1, n) / n * (1 - xSum);
    numIter = numIter + 1;
    if numIter > maxIter
        break
    end
end
pr = x;
end

