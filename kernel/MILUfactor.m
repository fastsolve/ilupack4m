function [M, options, prec] = MILUfactor(varargin)
%MILUfactor Perform multilevel-ILU factorization
%
%    M = MILUfactor(A) performs ILU factorization and returns an opaque
%    structure of the preconditioner. Matrix A can be in MATLAB's built-in 
%    sparse format or in CRS format created using crs_matrix.
%
%    M = MILUfactor(rowptr, colind, vals) takes a matrix in the CRS 
%    format instead of MATLAB's built-in sparse format.
%
%    M = MILUfactor(A, opts)
%    M = MILUfactor(rowptr, colind, vals, opts)
%    allows you to specify additional options for ILUPACK. see ILUinit
%    for additional options.
%
%    [M, options] = MILUfactor(...) returns an options structure in
%    addition to the preconditioner.
%
%    [M, options, prec] = MILUfactor(...) returns an options structure in
%    addition to the preconditioner.

if nargin == 0
    help MILUfactor
    return;
end

if issparse(varargin{1})
    A = varargin{1};
    next_index = 2;
elseif isstruct(varargin{1})
    A = crs_2sparse(varargin{1}.row_ptr, varargin{1}.col_ind, varargin{1}.val);
    next_index = 2;
else
    A = crs_2sparse(varargin{1}, varargin{2}, varargin{3});
    next_index = 4;
end

options = ILUinit(A);

if nargin >= next_index && ~isempty(varargin{next_index})
    opts = varargin{next_index};
    names = fieldnames(opts);
    for i = 1:length(names)
        options.(names{i}) = cast(opts.(names{i}), class(options.(names{i})));
        if isequal(names{i}, 'droptol') && ~isfield(opts, 'droptolS')
            options.droptolS = options.droptol * 0.1;
        end
    end
end

%% Perform ILU factorization
[prec, options] = ILUfactor(A, options);

%% Compute M(i).q and change M(i).U to incorporate D
M = repmat(struct(), length(prec), 1);
for i = 1:length(prec)
    M(i).p = int32(prec(i).p(:));
    M(i).q(prec(i).invq) = int32(1:prec(i).n);
    M(i).q = M(i).q(:);

    M(i).rowscal = prec(i).rowscal(:);
    M(i).colscal = prec(i).colscal(:);

    if ~issparse(prec(i).L)
        % Save L and U into U as a dense matrix
        LU = tril(prec(i).L, -1) + prec(i).D * prec(i).U;
        M(i).L = ccs_matrix(prec(i).n, prec(i).n);
        M(i).U = ccs_matrix(prec(i).n, prec(i).n);
        M(i).U.val = LU(:);
        M(i).d = zeros(0, 1);
    else
        % Extract strictly lower and upper triangular parts of L and U
        % Store transpose to allow parallelism
        M(i).L = ccs_createFromSparse(tril(prec(i).L, -1) / prec(i).D);
        M(i).U = ccs_createFromSparse(triu(prec(i).D \ prec(i).U, 1));
        M(i).d = diag(prec(i).D);
    end
    M(i).negE = crs_createFromSparse(-prec(i).E);
    M(i).negF = crs_createFromSparse(-prec(i).F);
end

if nargout < 3
    prec = ILUdelete(prec);
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
%! scaledA = diag(M(1).rowscal)*A*diag(M(1).colscal);
%! scaledA = scaledA(M(1).p, M(1).q);
%! 
%! if length(M)==1
%!     fprintf(1, 'M has one structure.\n');
%! elseif length(M)==2
%!     fprintf(1, 'M has two structures.\n');
%!     B = scaledA(1:prec(1).nB, 1:prec(1).nB);
%!     E = scaledA(prec(1).nB+1:prec(1).n, 1:prec(1).nB);
%!     F = scaledA(1:prec(1).nB, prec(1).nB+1:prec(1).n);
%!     C = scaledA(prec(1).nB+1:prec(1).n, prec(1).nB+1:prec(1).n);
%!     assert(max(max(abs(prec(1).E - E))) < droptol);
%!     assert(max(max(abs(prec(1).F - F))) < droptol);
%!     S = C - E * inv(B) * F;
%!     scaledS = diag(prec(2).rowscal) * S * diag(prec(2).colscal);
%!     q2(prec(2).invq) = 1:prec(2).n;
%!     if prec(2).n > 1
%!        assert(max(max(abs(scaledS(prec(2).p, q2) - prec(2).L * prec(2).D * prec(2).U))) < droptol * 50);
%!     end
%! end
%!
%! prec = ILUdelete(prec);

end
