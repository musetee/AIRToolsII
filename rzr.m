function [A,b] = rzr(A,b)
    A = A(any(A,2),:);
    b = b(any(b,2),:);