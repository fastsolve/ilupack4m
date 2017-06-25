function load_milu(varargin)

% ilupack4m depends on paracoder for testing. Need to load it first
if ~exist('load_m2c.m', 'file')
    if exist('../paracoder/load_m2c.m', 'file')
        run('../paracoder/load_m2c.m')
    end
elseif ~exist('m2c.m', 'file')
    load_m2c
end

% Load the Petsc4m module
addpath(miluroot); %#ok<*MCAP>
addpath([miluroot '/matlab/ilupack']);

end