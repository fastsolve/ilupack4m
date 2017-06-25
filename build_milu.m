function build_milu(varargin)
% Build ILUPACK

if ~exist('OCTAVE_VERSION', 'builtin') || nargin>=1 && isequal(varargin{1}, '-matlab')
  fprintf(1, 'Building for MATLAB...\n')
  system(['cd ' miluroot '/makefiles; make TARGET=MATLAB']);
else
  fprintf(1, 'Building for Octave...\n')
  system(['cd ' miluroot '/makefiles; make TARGET=Octave']);
end

end
