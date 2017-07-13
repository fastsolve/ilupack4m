function mat = MILU_Dmat(varargin) %#codegen
%MILU_Dmat Map an opaque object into a DAMGlevelmat pointer
%
%  MILU_Dmat() simply returns a definition of the m2c_opaque_type,
%  suitable in the argument specification for codegen.
%
%  MILU_Dmat(ptr) or MILU_Dmat(ptr, false) converts a given object to
%  a pointer to DAMGlevelmat.
%
%  MILU_Dmat(ptr, true) wraps a pointer into an opaque object. This should 
%  be used if the opaque object needs to be returned to MATLAB.

coder.inline('always');
coder.cinclude('ilupack.h');

mat = m2c_opaque_obj('DAMGlevelmat *', varargin{:});
