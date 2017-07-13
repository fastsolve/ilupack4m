function [x, flag, iter, resids, times] = fgmresMILU(varargin)
% fgmresMILU Solves a sparse system using FGMRES with MILU as preconditioner
%
% Syntax:
%    x = fgmresMILU(A, b) solves a sparse linear system using ILUPACK's
%    multilevel ILU as the right preconditioner. Matrix A can be in MATLAB's
%    built-in sparse format or in CRS format created using crs_matrix.
%
%    x = fgmresMILU(rowptr, colind, vals, b) takes a matrix in the CRS
%    format instead of MATLAB's built-in sparse format.
%
%    x = fgmresMILU(A, b, restart)
%    x = fgmresMILU(rowptr, colind, vals, b, restart)
%    allows you to specify the restart parameter for GMRES. The default
%    value is 30. You can preserve the default value by passing [].
%
%    x = fgmresMILU(A, b, restart, rtol, maxiter)
%    x = fgmresMILU(rowptr, colind, vals, b, restart, rtol, maxiter)
%    allows you to specify the relative tolerance and the maximum number
%    of iterations. Their default values are 1.e-5 and 10000, respectively.
%    Use 0 or [] to preserve default values of rtol and maxiter.
%
%    x = fgmresMILU(A, b, restart, rtol, maxiter, x0)
%    x = fgmresMILU(rowptr, colind, vals, b, restart, rtol, maxiter, x0)
%    takes an initial solution in x0. Use 0 or [] to preserve the default
%    initial solution (all zeros).
%
%    x = fgmresMILU(A, b, restart, rtol, maxit, x0, nthreads)
%    x = fgmresMILU(rowptr, colind, vals, b, restart, rtol, maxit, x0, nthreads, opts)
%    allows you to specify number of threads for matrix-vector multiplication
%
%    x = fgmresMILU(A, b, restart, rtol, maxit, x0, nthreads, opts)
%    x = fgmresMILU(rowptr, colind, vals, b, restart, rtol, maxit, x0, nthreads, opts)
%    allows you to specify additional options for ILUPACK.
%
%    [x, flag, iter, resids, times] = fgmresMILU(...) returns the iteration
%      counta, nd the history of relative residuals, and runtimes
%
% Note: The algorithm uses Householder reflectors for orthogonalization.
% It is more expensive than modified Gram-Schmidtz but is more robust.

if nargin == 0
    help fgmresMILU
    return;
end

if issparse(varargin{1})
    A = crs_matrix(varargin{1});
    next_index = 2;
elseif isstruct(varargin{1})
    A = varargin{1};
    next_index = 2;
else
    A = crs_matrix(varargin{1}, varargin{2}, varargin{3});
    next_index = 4;
end

if nargin < next_index
    error('The right hand-side must be specified');
else
    b = varargin{next_index};
end

verbose = int32(5 - nargout);

if verbose
    fprintf(1, 'Performing ILU facotirzation...\n');
end

% Perform ILU factorization
times = zeros(2, 1);
tic;
prec = MILUinit(varargin{1:next_index-1}, varargin{next_index+6:end});
times(1) = toc;

if verbose
    fprintf(1, 'Finished ILU factorization in %.1f seconds \n', times(1));
end

if nargin >= next_index + 1 && ~isempty(varargin{next_index+1})
    restart = int32(varargin{next_index+1});
else
    restart = int32(30);
end

if restart > 100
    m2c_warning('You set restart to %d. It is recommended to maker it no greater than 100.\n', restart);
end

if nargin >= next_index + 2 && ~isempty(varargin{next_index+2})
    rtol = double(varargin{next_index+2});
else
    rtol = 1.e-5;
end

if nargin >= next_index + 3 && ~isempty(varargin{next_index+3})
    maxit = int32(varargin{next_index+3});
else
    maxit = int32(10000);
end

if nargin >= next_index + 4 && ~isempty(varargin{next_index+4})
    x0 = varargin{next_index+4};
else
    x0 = cast([], class(b));
end

if nargin >= next_index + 5 && ~isempty(varargin{next_index+5})
    nthreads = int32(varargin{next_index+5});
else
    nthreads = ompGetMaxThreads();
end

if verbose
    fprintf(1, 'Starting Krylov solver ...\n');
end

tic;
if exist(['fgmresMILU_kernel.' mexext], 'file')
    % Calling MEX function
    ptr = MILU_Dmat(prec(1).ptr, true);
    param = MILU_Dparam(prec(1).param, true);

    [x, flag, iter, resids] = fgmresMILU_kernel(A, b, ptr, ...
        restart, rtol, maxit, x0, verbose, nthreads, param, ...
        prec(1).rowscal', prec(1).colscal');
else
    [x, flag, iter, resids] = fgmresMILU_kernel(A, b, prec, ...
        restart, rtol, maxit, x0, verbose, nthreads);
end
times(2) = toc;

if verbose
    fprintf(1, 'Finished solve in %d iterations and %.1f seconds.\n', iter, times(2));
end

prec = ILUdelete(prec); %#ok<NASGU>

end

function test %#ok<DEFNU>
%!test
%!shared A, b, rtol
%! system('gd-get -O -p 0ByTwsK5_Tl_PemN0QVlYem11Y00 fem2d"*".mat');
%! s = load('fem2d_cd.mat');
%! A = s.A;
%! s = load('fem2d_vec_cd.mat');
%! b = s.b;
%! rtol = 1.e-5;
%
%! [x, flag, iter, resids] = fgmresMILU(A, b, [], rtol, 100);
%! assert(norm(b - A*x) < rtol * norm(b))

end
