A = sprand(10, 10, 0.4);
b = A * ones(10, 1);

x = gmresMILU(A, b);
