function [x, flag, iter, resids, times] = bicgstabMILU(varargin)
% bicgstabMILU BiCGSTAB with MILU as right preconditioner
%
%    x = bicgstabMILU(A, b) solves a sparse linear system using ILUPACK's
%    multilevel ILU as the right preconditioner. Matrix A can be in MATLAB's
%    built-in sparse format or in CRS format created using crs_matrix.
%
%    x = bicgstabMILU(rowptr, colind, vals, b) takes a matrix in the CRS
%    format instead of MATLAB's built-in sparse format.
%
%    x = bicgstabMILU(A, b, rtol)
%    x = bicgstabMILU(rowptr, colind, vals, b, rtol)
%    specifies the relative tolerance and the maximum number of iterations.
%    If rtol is [], it will use the default value 1.e-6.
%
%    x = bicgstabMILU(A, b, rtol, maxit)
%    x = bicgstabMILU(rowptr, colind, vals, b, rtol, maxit)
%    specifies the maximum number of iterations. If maxit is [], it
%    will use the default value 500.
%
%    x = bicgstabMILU(A, b, rtol, maxiter, x0)
%    x = bicgstabMILU(rowptr, colind, vals, b, rtol, maxiter, x0)
%    takes an initial guess for x in x0. Use [] to preserve the default
%    initial solution (all zeros).
%
%    x = bicgstabMILU(A, b, ..., 'name', value, ...)
%    x = bicgstabMILU(rowptr, colind, vals, b, ..., 'name', value, ...)
%    allows omitting none or some of the positional arguments rtol,
%    maxiter and x0 and specifying these and other parameters in the form
%    'param1_name', param1_value, 'param2_name', param2_value, and so on.
%    The parameter names are not case sensitive. Available parameters and
%    their default values (enclosed by '[' and ']') are as follows:
%
%   'rtol' [1.e-6]:   Relative tolerance for converegnce
%
%   'maxiter' [500]:  Maximum number of iterations
%
%   'x0' [all-zeros]: Initial guess vector
%
%   'verb' [1]:  Verbosity level.
%          0 - silent
%          1 - iteration info every 30 iterations
%          2 - iteration info for all iterations
%
%   'ordering' ['amd']: Reorderings based on |A|+|A|'.
%          'amd'    - Approximate Minimum Degree
%          'metisn' - METIS multilevel nested dissection by NODES
%          'metise' - METIS multilevel nested dissection by EDGES
%          'rcm'    - Reverse Cuthill-McKee
%          'mmd'    - Minimum Degree
%          'amf'    - Approximate Minimum Fill
%          ''       - no reordering
%
%   'condest'  [5]: bound for the inverse triangular factors from the ILU
%
%   'droptol' [0.001]: threshold for dropping small entries during the
%    computation of the ILU factorization
%
%   'droptols' [droptol*0.1]: threshold for dropping small entries from the
%    Schur complement
%
%   'lfil' [unlimited]: restrict the number of nonzeros per column in L
%    (and respectively per row in U) to at most 'lfil' entries.
%
%   'lfils' [unlimited]: restrict the number of nonzeros per row in the
%    approximate Schur complement to at most 'lfilS' entries.
%
%   'elbow' [10]: Elbow space for memory of the ILUPACK multilevel
%    preconditioner as estimation of maximum number of fills as the
%    initial matrix.
%
%   'nthreads' [1]: maximal number of threads to use
%
%    [x, flag] = bicgstabMILU(...) returns a convergence flag.
%    flag  0 - solution found to tolerance
%          1 - no convergence given max_it
%         -1 - breakdown: rho = 0
%         -2 - breakdown: omega = 0
%         -3 - divergence (relative tolerance > 100)
%
%    [x, flag, iter] = bicgstabMILU(...) returns the iteration count.
%
%    [x, flag, iter, resids] = bicgstabMILU(...) returns the relative
%    residual in 2-norm at each iteration.
%
%    [x, flag, iter, resids, times] = bicgstabMILU(...) returns the setup
%    time (times(1)) and solve time (times(2)) in seconds.
%
%  See also bicgstabMILU

if nargin == 0
    help bicgstabMILU
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

% Initialize default arguments
verbose = int32(1);
rtol = 1.e-6;
maxit = int32(500);
x0 = cast([], class(b));
nthreads = int32(1);

params_start = nargin;
for i = next_index+1:nargin
    if ischar(varargin{i})
        params_start = i;
        break
    end
end

% Process positional arguments
if params_start > next_index + 1 && ~isempty(varargin{next_index+1})
    rtol = double(varargin{next_index+1});
end

if params_start > next_index + 2 && ~isempty(varargin{next_index+2})
    maxit = int32(varargin{next_index+2});
end

if params_start > next_index + 3 && ~isempty(varargin{next_index+3})
    x0 = varargin{next_index+3};
end

% Process argument-value pairs to update arguments
options = struct('ordering', 'amd', 'droptol', 0.001, ...
    'condest', 5, 'elbow', 10, 'lfil', size(b, 1), 'lfilS', size(b, 1));
for i = params_start:2:length(varargin)-1
    switch lower(varargin{i})
        case {'maxit', 'maxiter'}
            maxit = int32(varargin{i+1});
        case 'x0'
            x0 = varargin{i+1};
        case {'rtol', 'reltol'}
            rtol = varargin{i+1};
        case {'verb', 'verbose'}
            verbose = int32(varargin{i+1});
        case 'nthreads'
            nthreads = int32(varargin{i+1});
        case 'ordering'
            options.ordering = varargin{i+1};
        case {'droptol', 'condest', 'elbow', 'lfil'}
            options.(lower(varargin{i})) = double(varargin{i+1});
        case 'droptols'
            options.droptolS = double(varargin{i+1});
        case 'lfils'
            options.lfilS = double(varargin{i+1});
        otherwise
            error('Unknown tuning parameter "%s"', varargin{i});
    end
end

if ~isfield(options, 'droptolS')
    options.droptolS = options.droptol * 0.1;
end

if verbose
    fprintf(1, 'Performing ILU facotirzation...\n');
end

% Perform ILU factorization
times = zeros(2, 1);
tic;
[prec, newoptions] = MILUinit(varargin{1:next_index-1}, options);
times(1) = toc;

if verbose
    if newoptions.elbow < 1
        warning('The number of fills is about %.1f%% of original matrix. You may want to decrease droptol to %g.\n', ...
            newoptions.elbow*100, options.droptol*0.1);
    else
        fprintf(1, 'The number of fills is about %.1f%% of original matrix.\n', ...
            newoptions.elbow*100);
    end
    fprintf(1, 'Finished ILU factorization in %.1f seconds \n', times(1));
end

if verbose
    fprintf(1, 'Starting Krylov solver ...\n');
end

tic;
if exist(['bicgstabMILU_kernel.' mexext], 'file')
    % Calling MEX function
    ptr = MILU_Dmat(prec(1).ptr, true);
    param = MILU_Dparam(prec(1).param, true);

    [x, flag, iter, resids] = bicgstabMILU_kernel(A, b, ptr, ...
        rtol, maxit, x0, verbose, nthreads, param, ...
        prec(1).rowscal', prec(1).colscal');
else
    [x, flag, iter, resids] = bicgstabMILU_kernel(A, b, prec, ...
        rtol, maxit, x0, verbose, nthreads);
end
times(2) = toc;

if verbose
    if flag == 0
        fprintf(1, 'Finished solve in %d iterations and %.1f seconds.\n', iter, times(2));
    elseif flag == 3
        fprintf(1, 'BiCGSTAB diverged after %d iterations and %.1f seconds.\n', iter, times(2));
    else
        fprintf(1, 'BiCGSTAB failed to converge after %d iterations and %.1f seconds.\n', iter, times(2));
    end
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
%! [x, flag, iter, resids] = bicgstabMILU(A, b,  rtol, 100);
%! assert(norm(b - A*x) <= rtol * norm(b))
%!
%!test
%! [x, flag, iter, resids] = bicgstabMILU(A, b, 'rtol', rtol, ...
%!         'maxit', 100);
%! assert(norm(b - A*x) <= rtol * norm(b))

end
