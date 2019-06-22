function build_milu(varargin)
% Build ILUPACK

if ~exist('OCTAVE_VERSION', 'builtin') || nargin >= 1 && isequal(varargin{1}, '-matlab')
    fprintf(1, 'Building for MATLAB with MUMPS...\n')
    system(['cd ', miluroot, '/makefiles; make -f makefile_mc64 TARGET=MATLAB']);

    LIBDIR = [miluroot, '/lib/GNU64_long'];
else
    fprintf(1, 'Building for Octave with MC64...\n')
    system(['cd ', miluroot, '/makefiles; make -f makefile TARGET=Octave']);
    LIBDIR = [miluroot, '/lib/GNU64'];
end

m2c('-mex', '-omp', '-O3', varargin{:}, ['-I', miluroot, '/include'], ...
    ['-L', LIBDIR], '-lilupack', 'MILUsolve');
m2c('-mex', '-omp', '-O3', varargin{:}, ['-I', miluroot, '/include'], ...
    ['-L', LIBDIR], '-lilupack', 'gmresMILU_HO');
m2c('-mex', '-omp', '-O3', varargin{:}, ['-I', miluroot, '/include'], ...
    ['-L', LIBDIR], '-lilupack', 'gmresMILU_MGS');
m2c('-mex', '-omp', '-O3', varargin{:}, ['-I', miluroot, '/include'], ...
    ['-L', LIBDIR], '-lilupack', 'gmresMILU_CGS');
m2c('-mex', '-omp', '-O3', varargin{:}, ['-I', miluroot, '/include'], ...
    ['-L', LIBDIR], '-lilupack', 'bicgstabMILU_kernel');

end
