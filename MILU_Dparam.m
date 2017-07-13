function ptr = MILU_Dparam(varargin) %#codegen
%MILU_Dparam Map an opaque object into a DILUPACKparam pointer
%
%  MILU_Dparam() simply returns a definition of the m2c_opaque_type,
%  suitable in the argument specification for codegen.
%
%  MILU_Dparam(ptr) or MILU_Dptr(ptr, false) converts a given object to
%  a pointer to DILUPACKparam.
%
%  MILU_Dparam(ptr, true) wraps a pointer into an opaque object. This should 
%  be used if the opaque object needs to be returned to MATLAB.

coder.inline('always');
coder.cinclude('ilupack.h');

ptr = m2c_opaque_obj('DILUPACKparam *', varargin{:});
