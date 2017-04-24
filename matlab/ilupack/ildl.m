function [L,D,P,scal,options]=ildl(A,options)
% [L,D,P,S]=ildl(A)
% compute symmetric indefinite and incomplete LDL^T factorization using the
% default options
%
% This routine approximately computes P^T S A S P ~  L D L^T,
% where S is a real diagonal (scaling) matrix, P is a permutation matrix,
% L is lower unit triangular matrix and D is a block diagonal matrix
% with 1x1 and 2x2 diagonal blocks
%
% [L,D,P,S]=ildl(A,options)
% same routine using your private options
%
% [L,D,P,S,options]=ildl(A,options)
% same routine returning the options used and possible warnings/error codes
%
%
% Input
% -----
% A           real symmetric, complex Hermitian or complex symmetric matrix
% options     optional options. If not passed, default options are used
%             1) options.matching:        use maximum weight matching prior
%                                         to any reordering or factorization
%                                         default 1 (i.e., turned on)
%             2) options.ordering         reorder the system according to
%                                         some symmetric reordering. This
%                                         is performed preserving the structure
%                                         obtained by maximum weight matching
%                                         'metisn' nested dissection by nodes
%                                                  (default)
%                                         'metise' nested dissection by edges
%                                         'amd' approximate mininum degree
%                                         'rcm' reverse Cuthill-McKee
%                                         'none' (only available if matching is
%                                                 turned off)
%            3) options.droptol           threshold for ILU, default 1e-3
%            4) options.diagcomp          diagonal compensation, default 0 (off)
%            5) options.lfil              maximum number of off-diagonal entries 
%                                         per row, default 0 (turned off)
%            6) options.ind               indicator array of length n, negative
%                                         entries indicate second block in a
%                                         Stokes-type block system
%
%
% Output
% -----
% L          lower triangular matrix with unit diagonal
% D          block diagonal matrix
% P          permutation matrix
% S          real diagonal scaling matrix
% options    optional output, see input

myoptions.matching=1;
myoptions.ordering='metisn';
myoptions.droptol=1e-3;
myoptions.diagcomp=0;
myoptions.lfil=size(A,1)+1;


if nargin==2
   if isfield(options,'matching')
      myoptions.matching=options.matching;
   end
   if isfield(options,'ordering')
      myoptions.ordering=options.ordering;
      if options.matching && strcmp(options.ordering,'none')
	 myoptions.ordering='metisn';
      end
      if (   ~strcmp(options.ordering,'metisn') ...
	  && ~strcmp(options.ordering,'metise') ...
          && ~strcmp(options.ordering,'amd') ...
	  && ~strcmp(options.ordering,'rcm') ...
	  && ~strcmp(options.ordering,'none'))
	 myoptions.ordering='metisn';
      end     
   end
   if isfield(options,'droptol')
      myoptions.droptol =options.droptol;
   end
   if isfield(options,'diagcomp')
      myoptions.diagcomp=options.diagcomp;
   end
   if isfield(options,'lfil')
      if options.lfil>0
	 myoptions.lfil=options.lfil;
      end
   end
   if isfield(options,'ind')
      if length(options.ind)==size(A,1)
	 myoptions.ind=options.ind;
      end
   end
end

if norm(A-A.',1)==0
   if isreal(A)
      [L,D,P,scal,myoptions]=DSYMildlfactor(A,myoptions);
   else
      [L,D,P,scal,myoptions]=ZSYMildlfactor(A,myoptions);
   end % if-else
elseif norm(A-A',1)==0
   if isreal(A)
      [L,D,P,scal,myoptions]=DSYMildlfactor(A,myoptions);
   else
      [L,D,P,scal,myoptions]=ZHERildlfactor(A,myoptions);
    end % if-else
else
   fprintf('matrix must be symmetric\n');
   return;
end % if-else
 
options.matching=myoptions.matching;
options.ordering=myoptions.ordering;
options.droptol =myoptions.droptol;
if isfield(options,'lfil')
   if options.lfil>0
      options.lfil=myoptions.lfil;
   else
      options.lfil=0;
   end
else
   options.lfil=0;
end
