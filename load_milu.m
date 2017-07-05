function load_milu(varargin)

persistent loaded;

% Load only once
if ~isempty(loaded)
  return
else
  loaded = true;
end

% ilupack4m depends on paracoder for testing. Load it first
if ~exist('load_m2c.m', 'file')
    if exist('../paracoder/load_m2c.m', 'file')
        run('../paracoder/load_m2c.m')
    end
elseif ~exist('m2c.m', 'file')
    load_m2c
end

% Load ILUPACK
addpath(miluroot); %#ok<*MCAP>
addpath([miluroot '/matlab/ilupack']);
addpath([miluroot '/util']);

% Show help message
if ~exist(['DGNLilupackinit.' mexext], 'file')
    disp('Please run build_milu to build ILUPACK.');
end

end
