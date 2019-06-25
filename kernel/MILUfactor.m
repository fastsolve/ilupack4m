function [M, options, prec, runtime] = MILUfactor(varargin)
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

if nargin >= next_index && ~isempty(varargin{next_index})
    opts = varargin{next_index};
    options = ILUinit(A, opts);
    names = fieldnames(opts);
    for i = 1:length(names)
        if isfield(options, names{i})
            options.(names{i}) = cast(opts.(names{i}), class(options.(names{i})));
        else
            options.(names{i}) = opts.(names{i});
        end
        if isequal(names{i}, 'droptol') && ~isfield(opts, 'droptolS')
            options.droptolS = options.droptol * 0.1;
        end
    end
else
    options = ILUinit(A);
end

%% Perform ILU factorization
tic
[prec, options] = ILUfactor(A, options);
runtime = toc;

nnz_total = 0;
nnz_offdiag = 0;  % nonzeros in off-diagonal blocks (i.e., E and F)
encountered_block_diag = 0;

%% Compute M(i).q and change M(i).U to incorporate D
M = repmat(struct(), length(prec), 1);
for i = 1:length(prec)
    if isempty(prec(i).U)
        nnz_total = nnz_total + 2*nnz(prec(i).L) - nnz(prec(i).D);
        nnz_offdiag = nnz_offdiag  + 2*nnz(prec(i).E);
    else
        nnz_total = nnz_total + nnz(prec(i).L) + nnz(prec(i).U) - nnz(prec(i).D);
        nnz_offdiag = nnz_offdiag + nnz(prec(i).E) + nnz(prec(i).F);
    end
    
    if nnz(prec(i).D) ~= prec(i).nB
        encountered_block_diag = 1;
    end
    if encountered_block_diag
        continue;
    end
    
    M(i).p = int32(prec(i).p(:));
    M(i).q(prec(i).invq) = int32(1:prec(i).n);
    M(i).q = M(i).q(:);

    M(i).rowscal = prec(i).rowscal(:);
    M(i).colscal = prec(i).colscal(:);
    
    if isequal(M(i).p, M(i).q) && isequal(M(i).rowscal, M(i).colscal)
        fprintf(1, 'Note: Level %d uses symmetric reordering and scaling for a nonsymmetric block.\n', i);
    elseif isequal(M(i).p, M(i).q) && ~all(M(i).p==(1:prec(i).n)')
        fprintf(1, 'Note: Level %d uses symmetric reordering but nonsymmetric scaling.\n', i);
    elseif isequal(M(i).rowscal, M(i).colscal) && ~all(M(i).rowscal==1)
        fprintf(1, 'Note: Level %d uses symmetric scaling but nonsymmetric reordering.\n', i);
    end

    if ~issparse(prec(i).L)
        % Save L and U into U as a dense matrix
        if isempty(prec(i).U)
            LU = tril(prec(i).L, -1) + prec(i).D * prec(i).L';
        else
            LU = tril(prec(i).L, -1) + prec(i).D * prec(i).U;
        end
        M(i).L = ccs_matrix(prec(i).nB, prec(i).nB);
        M(i).U = ccs_matrix(prec(i).nB, prec(i).nB);
        M(i).U.val = LU(:);
        M(i).d = zeros(0, 1);
    else
        % Extract strictly lower and upper triangular parts of L and U
        % Store transpose to allow parallelism
        M(i).L = ccs_createFromSparse(tril(prec(i).L, -1) / prec(i).D);
        if isempty(prec(i).U)
            M(i).U = ccs_createFromSparse((tril(prec(i).L, -1) / prec(i).D)');
        else
            M(i).U = ccs_createFromSparse(triu(prec(i).D \ prec(i).U, 1));
        end
        M(i).d = diag(prec(i).D);
    end
    
    M(i).negE = crs_createFromSparse(-prec(i).E);
    M(i).negF = crs_createFromSparse(-prec(i).F);
end

options.nnz_offdiag = nnz_offdiag;
options.nnz_total = nnz_total + nnz_offdiag;

if nargout < 3
    prec = ILUdelete(prec);
end

if encountered_block_diag
    warning('Encountered 2x2 diagonal blocks in ILU. Cannot convert PREC.');
    M = [];
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
