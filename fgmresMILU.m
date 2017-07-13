function [x, flag, iter, resids, times] = fgmresMILU(varargin)
% fgmresMILU GMRES with MILU as right preconditioner
%
%    x = fgmresMILU(A, b) solves a sparse linear system using ILUPACK's
%    multilevel ILU as the right preconditioner. Matrix A can be in MATLAB's
%    built-in sparse format or in CRS format created using crs_matrix.
%
%    x = fgmresMILU(rowptr, colind, vals, b) takes a matrix in the CRS
%    format instead of MATLAB's built-in sparse format.
%
%    x = fgmresMILU(A, b, restart)
%    x = fgmresMILU(rowptr, colind, vals, b, restart)
%    specifies the restart parameter for GMRES. The default value is 30.
%    You can preserve the default value by passing [].
%
%    x = fgmresMILU(A, b, restart, rtol, maxit)
%    x = fgmresMILU(rowptr, colind, vals, b, restart, rtol, maxit)
%    specifies the relative tolerance and the maximum number of iterations.
%    Their default values are 1.e-5 and 10000, respectively. Use 0 or [] to
%    preserve default values of rtol and maxiter.
%
%    x = fgmresMILU(A, b, restart, rtol, maxiter, x0)
%    x = fgmresMILU(rowptr, colind, vals, b, restart, rtol, maxiter, x0)
%    takes an initial guess for x in x0. Use 0 or [] to preserve the default
%    initial solution (all zeros).
%
%    x = fgmresMILU(A, b, ..., 'name', value, ...)
%    x = fgmresMILU(rowptr, colind, vals, b, ..., 'name', value, ...)
%    allows omitting none or some of the positional arguments restart, rtol,
%    maxiter and x0 and specifying these and other parameters in the form
%    'param1_name', param1_value, 'param2_name', param2_value, and so on.
%    The parameter names are not case sensitive. Available parameters and 
%    their default values (enclosed by '[' and ']') are as follows:
%
%   'restart' [30]:   Number of iterations before restart
%
%   'rtol' [1.e-6]:   Relative tolerance for converegnce
%
%   'maxiter' [1e5]:  Maximum number of iterations
%
%   'x0' [all-zeros]: Initial guess
%
%   'verb' [5-nargout]:  Verbosity level. 
%          0 - silent
%          1 - outer iteration info
%          2 - inner iteration info
%
%   'orth' ['GS']: Orthogonalization strategy. 
%          'GS' - modified Gauss-Seidel
%          'HO' - Householder
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
%   'condest'  [100]: bound for the inverse triangular factors from the ILU
%
%   'droptol' [0.01]: threshold for dropping small entries during the 
%    computation of the ILU factorization
%
%   'droptols' [0.01]: threshold for dropping small entries from the 
%    Schur complement
%
%   'lfil' [inf]: restrict the number of nonzeros per column in L 
%    (and respectively per row in U) to at most 'lfil' entries.
%
%   'lfils' [inf]: restrict the number of nonzeros per row in the 
%    approximate Schur complement to at most 'lfilS' entries.
%
%   'elbow' [10]: Elbow space for memory of the ILUPACK multilevel 
%    preconditioner as estimation of maximum number of fills as the 
%    initial matrix.
%
%   'nthreads' [omp_get_num_procs()]: maximal number of threads to use
%
%    [x, flag] = fgmresMILU(...) returns a convergence flag.
%    flag: 0 - converged to the desired tolerance TOL within MAXIT iterations.
%          1 - iterated maxit times but did not converge.
%          3  -stagnated (two consecutive iterates were the same).

%    [x, flag, iter] = fgmresMILU(...) returns the iteration count.
%
%    [x, flag, iter, resids] = fgmresMILU(...) returns the relative 
%    residual in 2-norm at each iteration.
%
%    [x, flag, iter, resids, times] = fgmresMILU(...) returns the setup
%    time (times(1)) and solve time (times(2)) in seconds.
%
%  See also bicgstabMILU

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

% Initialize default arguments
verbose = int32(5 - nargout);
rtol = 1.e-5;
maxit = int32(10000);
restart = int32(30);
x0 = cast([], class(b));
nthreads = ompGetNumProcs;
Householder = false;

params_start = nargin;
for i = next_index+1:nargin
    if ischar(varargin{i})
        params_start = i;
        break
    end
end

% Process positional arguments
if params_start > next_index + 1 && ~isempty(varargin{next_index+1})
    restart = int32(varargin{next_index+1});
end

if restart > 100
    m2c_warning('You set restart to %d. It is recommended to maker it no greater than 100.\n', restart);
end

if params_start > next_index + 2 && ~isempty(varargin{next_index+2})
    rtol = double(varargin{next_index+2});
end

if params_start > next_index + 3 && ~isempty(varargin{next_index+3})
    maxit = int32(varargin{next_index+3});
end

if params_start > next_index + 4 && ~isempty(varargin{next_index+4})
    x0 = varargin{next_index+4};
end

% Process argument-value pairs to update arguments
options = struct();
for i = params_start:2:length(varargin)-1
    switch lower(varargin{i})
        case {'maxit', 'maxiter'}
            maxit = int32(varargin{i+1});
        case {'restart', 'nrestart'}
            restart = int32(int32(varargin{i+1}));
        case 'x0'
            x0 = varargin{i+1};
        case {'rtol', 'reltol'}
            rtol = varargin{i+1};
        case {'verb', 'verbose'}
            verbose = int32(varargin{i+1});
        case 'orth'
            Householder = isequal(varargin{i+1}, 'HO');
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

if verbose
    fprintf(1, 'Performing ILU facotirzation...\n');
end

% Perform ILU factorization
times = zeros(2, 1);
tic;
prec = MILUinit(varargin{1:next_index-1}, options);
times(1) = toc;

if verbose
    fprintf(1, 'Finished ILU factorization in %.1f seconds \n', times(1));
end

if verbose
    fprintf(1, 'Starting Krylov solver ...\n');
end

if Householder
    kernel = 'fgmresMILU_HO';
else
    kernel = 'fgmresMILU_GS';
end  
kernel_func = eval(['@' kernel]);

tic;
if exist([kernel '.' mexext], 'file')
    % Calling MEX function
    ptr = MILU_Dmat(prec(1).ptr, true);
    param = MILU_Dparam(prec(1).param, true);

    [x, flag, iter, resids] = kernel_func(A, b, ptr, ...
        restart, rtol, maxit, x0, verbose, nthreads, param, ...
        prec(1).rowscal', prec(1).colscal');
else
    [x, flag, iter, resids] = kernel_func(A, b, prec, ...
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
%! [x, flag, iter, resids] = fgmresMILU(A, b, 'rtol', rtol, 'maxit', 100, 'ordering', 'amd');
%! assert(norm(b - A*x) <= rtol * norm(b))
%
%! [x, flag, iter, resids] = fgmresMILU(A, b, [], rtol, 100);
%! assert(norm(b - A*x) <= rtol * norm(b))

end
