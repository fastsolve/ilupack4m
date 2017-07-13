function [x, flag, iter, times] = gmresMILU(varargin)
% Solves a sparse system using GMRES with Multilevel ILU as right preconditioner
%
% Syntax:
%    x = gmresMILU(A, b) solves a sparse linear system using ILUPACK's
%    multilevel ILU as the right preconditioner. Matrix A can be in MATLAB's
%    built-in sparse format or in CRS format created using crs_matrix.
%
%    x = gmresMILU(rowptr, colind, vals, b) takes a matrix in the CRS
%    format instead of MATLAB's built-in sparse format.
%
%    x = gmresMILU(A, b, restart)
%    x = gmresMILU(rowptr, colind, vals, b, restart)
%    allows you to specify the restart parameter for GMRES. The default
%    value is 30. You can preserve the default value by passing [].
%
%    x = gmresMILU(A, b, restart, rtol, maxiter)
%    x = gmresMILU(rowptr, colind, vals, b, restart, rtol, maxiter)
%    allows you to specify the relative tolerance and the maximum number
%    of iterations. Their default values are 1.e-5 and 10000, respectively.
%    Use 0 or [] to preserve default values of rtol and maxiter.
%
%    x = gmresMILU(A, b, restart, rtol, maxiter, x0)
%    x = gmresMILU(rowptr, colind, vals, b, restart, rtol, maxiter, x0)
%    takes an initial solution in x0. Use 0 or [] to preserve the default
%    initial solution (all zeros).
%
%    x = gmresMILU(A, b, restart, rtol, maxit, x0, opts)
%    x = gmresMILU(rowptr, colind, vals, b, restart, rtol, maxit, x0, opts)
%    allows you to specify additional options for ILUPACK.
%
%    [x, flag, iter, times] = gmresMILU(...) returns the iteration
%      counts and runtimes.

if nargin == 0
    help gmresMILU
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

if nargin < next_index
    error('The right hand-side must be specified');
else
    b = varargin{next_index};
end

% Perform ILU factorization
times = zeros(2, 1);
tic;
[PREC, options] = MILUinit(varargin{1:next_index-1}, varargin{next_index+5:end});
times(1) = toc;

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

if nargin >= next_index + 5 && ~isempty(varargin{next_index+5})
    opts = varargin{next_index+5};
    names = fieldnames(opts);
    for i = 1:length(names)
        options.(names(i)) = cast(opts.(names(i)), class(options.(names(i))));
    end
end

if nargout < 4
    fprintf(1, 'Finished setup in %.1f seconds \n', times(1));
end

tic;
if isempty(x0)
    [x, options] = ILUsolver(A, PREC, options, b);
else
    [x, options] = ILUsolver(A, PREC, options, b, x0);
end
times(2) = toc;

PREC = ILUdelete(PREC); %#ok<NASGU>

if options.niter >= options.maxit
    % Reached max number of iterations
    flag = int32(3);
else
    flag = int32(0);
end

iter = options.niter;

if nargout < 4
    fprintf(1, 'Finished solve in %d iterations and %.1f seconds.\n', options.niter, times(2));
end

end

function test %#ok<DEFNU>
%!test
%!shared A, b, rtol
%! system('gd-get -O -p 0ByTwsK5_Tl_PemN0QVlYem11Y00 fem2d"*".mat');
%! s = load('fem2d_cd.mat');
%! A = s.A;
%! s = load('fem2d_vec_cd.mat');
%! b = s.b;
%! rtol = 1.e-7;
%
%! x = gmresMILU(A, b, [], rtol, 58);
%! assert(norm(b - A*x) < 1.e3 * rtol * norm(b))

%!test
%! x = gmresMILU(A, b, 30, rtol, 58);
%! assert(norm(b - A*x) < 1.e3 * rtol * norm(b))

end
