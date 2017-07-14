function [prec, options] = MILUinit(varargin)
%Initialize a multilevel-ILU preconditioner
%
%    prec = MILUinit(A) performs ILU factorization and returns an opaque
%    structure of the preconditioner. Matrix A can be in MATLAB's built-in 
%    sparse format or in CRS format created using crs_matrix.
%
%    prec = MILUinit(rowptr, colind, vals) takes a matrix in the CRS 
%    format instead of MATLAB's built-in sparse format.
%
%    prec = MILUinit(A, opts)
%    prec = MILUinit(rowptr, colind, vals, opts)
%    allows you to specify additional options for ILUPACK. see ILUinit
%    for additional options.
%
%    [prec, options] = MILUinit(...) returns an options structure in
%    addition to the preconditioner.

if nargin == 0
    help MILUinit
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

options = ILUinit(A);

if nargin >= next_index && ~isempty(varargin{next_index})
    opts = varargin{next_index};
    names = fieldnames(opts);
    for i = 1:length(names)
        options.(names{i}) = cast(opts.(names{i}), class(options.(names{i})));
    end
end

% Perform ILU factorization
[prec, options] = ILUfactor(A, options);

end
