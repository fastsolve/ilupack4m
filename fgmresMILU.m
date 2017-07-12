function [x, iter, resids, times] = fgmresMILU(varargin)
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
%    x = fgmresMILU(A, b, restart, rtol, maxit, x0, opts)
%    x = fgmresMILU(rowptr, colind, vals, b, restart, rtol, maxit, x0, opts)
%    allows you to specify additional options for ILUPACK.
%
%    [x, iter, resids, times] = fgmresMILU(...) returns the iteration count
%      and the history of relative residuals, and runtimes in addition to x
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

% Perform ILU factorization
times = zeros(2, 1);
[prec, options, times(1)] = MILUinit(varargin{1:next_index-1}, varargin{next_index+1:end});
if nargout < 3
    fprintf(1, 'Finished setup in %.1f seconds \n', times(1));
end

if nargin >= next_index + 1 && ~isempty(varargin{next_index+1})
    options.nrestart = cast(varargin{next_index+1}, class(options.nrestart));
else
    options.nrestart = cast(30, class(options.nrestart));
end

if nargin >= next_index + 2 && ~isempty(varargin{next_index+2})
    options.restol = cast(varargin{next_index+2}, class(options.restol));
else
    options.restol = cast(1.e-5, class(options.restol));
end

if nargin >= next_index + 3 && ~isempty(varargin{next_index+3})
    options.maxit = cast(varargin{next_index+3}, class(options.maxit));
else
    options.maxit = cast(10000, class(options.maxit));
end

if nargin >= next_index + 4 && ~isempty(varargin{next_index+4})
    x0 = varargin{next_index+4};
else
    x0 = cast([], class(b));
end

tic;
[x, iter, resids] = fgmresMILU_kernel(A, b, prec, x0, options);
times(2) = toc;

end
