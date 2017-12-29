function [b, y1, y2] = MILUsolve(M, b, y1, y2)
%MILUsolve computes M\b, where M is the preconditioner
%   b = MILUsolve(M, b)
%   M is a structure containing the multilevel ILU factorization of A.
%
%   [b, y1, y2] = MILUsolve(M, b, y1, y2)
%   where y1 and y2 are size n buffers.
%
%   At each level of M, L * U is equal to 
%   the nB-b-nB leadng block of
%     P * diag(rowscal) * A * diag(colcale) * Q
%   In the coarsest level, if the matrix is nearly dense, then 
%   tril(L, -1) + U are stored together as a dense matrix in U.val

%#codegen -args {MILU_Prec, m2c_vec, m2c_vec, m2c_vec}
%#codegen MILUsolve_2args -args {MILU_Prec, m2c_vec}

zero = coder.ignoreConst(int32(0));
one = coder.ignoreConst(int32(1));

if nargin<3
    y1 = zeros(max(M(1).L.nrows, M(1).negE.nrows), 1);
end
if nargin<4
    y2 = zeros(M(1).negE.nrows, 1);
end

[b, y1, y2] = solve_milu(M, one, b, zero, y1, y2);

end

function [b, y1, y2] = solve_milu(M, lvl, b, offset, y1, y2)
coder.inline('never');

nB = M(lvl).L.nrows;
n = nB + M(lvl).negE.nrows;

% Rescale and permute first block of b
for i = 1:nB
    k = M(lvl).p(i);
    y1(i) = M(lvl).rowscal(k) .* b(k + offset);
end
% Rescale and permute second block of b
for i = (nB + 1):n
    k = M(lvl).p(i);
    y2(i-nB) = M(lvl).rowscal(k) .* b(k + offset);
end

if n > nB
    for i = 1:nB
        b(offset + i) = y1(i);
    end
end

if isempty(M(lvl).L.val) && numel(M(lvl).U.val) == n * n
    % L is empty and U is a dense matrix storing result from dgetrf
    y1 = solve_getrs(M(lvl).U.val, y1, nB);
else
    % It only accesses the first nB entries
    y1 = ccs_solve_utril(M(lvl).L, y1);
    for i = 1:nB
        y1(i) = y1(i) / M(lvl).d(i);
    end
    y1 = ccs_solve_utriu(M(lvl).U, y1);
end

if n > nB
    y2 = crs_Axpy(M(lvl).negE, y1, y2);
    for i = 1:n-nB
        b(offset + nB + i) = y2(i);
    end

    [b, y1, y2] = solve_milu(M, lvl+1, b, offset + nB, y1, y2);

    for i = 1:nB
        y1(i) = b(offset + i);
    end
    for i = 1:n-nB
        y2(i) = b(offset + nB + i);
    end

    y1 = crs_Axpy(M(lvl).negF, y2, y1);
    y1 = ccs_solve_utril(M(lvl).L, y1);
    for i = 1:nB
        y1(i) = y1(i) / M(lvl).d(i);
    end
    y1 = ccs_solve_utriu(M(lvl).U, y1);
end

% Rescale and permute solution vector
for i = 1:nB
    k = M(lvl).q(i);
    b(k + offset) = y1(i) * M(lvl).colscal(k);
end
for i = (nB + 1):n
    k = M(lvl).q(i);
    b(k + offset) = y2(i-nB) * M(lvl).colscal(k);
end

end

function test %#ok<DEFNU>
%!test
%! n = 10;
%! density = 0.4;
%! droptol = 0.001;
%!
%! for i=1:100
%!     A = sprand(n, n, density);
%!     if condest(A) < 1e4
%!         break;
%!     end
%! end
%! save -v7 random_mat.mat A
%! b = A * ones(n, 1);
%!
%! [M, ~, prec] = MILUfactor(A, struct('droptol', droptol));
%!
%! x_ref = ILUsol(prec, b);
%! x = MILUsolve(M, b);
%! assert(norm(x - x_ref) < 1.e-8);
%! prec = ILUdelete(prec);

%!test
%!shared A, b, rtol
%! system('gd-get -O -p 0ByTwsK5_Tl_PemN0QVlYem11Y00 fem2d"*".mat');
%! s = load('fem2d_cd.mat');
%! A = s.A;
%! s = load('fem2d_vec_cd.mat');
%! b = s.b;
%!
%! fprintf('Computing ILU factorization...'); tic;
%! [M, ~, prec] = MILUfactor(A, struct('droptol', 0.001));
%! fprintf('Done in %g seconds\n', toc);
%!
%! fprintf('Calling ILUsol...'); tic;
%! x_ref = ILUsol(prec, b);
%! fprintf('Done in %g seconds\n', toc);

%! fprintf('Calling MILUsolve...'); tic;
%! x = MILUsolve(M, b);
%! fprintf('Done in %g seconds\n', toc);

%! assert(norm(x - x_ref, inf) < 1.e-8);
%! prec = ILUdelete(prec);

end
