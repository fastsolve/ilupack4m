function root = miluroot
% PETSCROOT  Determine root directory of ilupack4m

persistent root__;

if isempty(root__)
    root__ = fileparts(which('miluroot.m'));
end

root = root__;
