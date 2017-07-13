function build_milu(varargin)
% Build ILUPACK

if ~exist('OCTAVE_VERSION', 'builtin') || nargin >= 1 && isequal(varargin{1}, '-matlab')
    fprintf(1, 'Building for MATLAB...\n')
    system(['cd ', miluroot, '/makefiles; make TARGET=MATLAB']);

    m2c('-mex', '-omp', '-O2', varargin{:}, ['-I', miluroot, '/include'], ...
        ['-L', miluroot, '/lib/GNU64_long'], '-lilupack', 'fgmresMILU_kernel');
else
    fprintf(1, 'Building for Octave...\n')
    system(['cd ', miluroot, '/makefiles; make TARGET=Octave']);

    m2c('-mex', '-omp', '-O2', varargin{:}, ['-I', miluroot, '/include'], ...
        ['-L', miluroot, '/lib/GNU64'], '-lilupack', 'fgmresMILU_kernel');
end


end
