% Startup script for petsc4m

if exist('OCTAVE_VERSION', 'builtin')
    more off;
else
    warning('off', 'MATLAB:mex:GccVersion_link');
end
load_milu;

load_milu;
