function [p] = symcamd(A,c)
% [p] = symcamd(A,c)
% 
% reorder a given nxn SYMMETRIC/HERMITIAN matrix A using constrained
% approximate minimum degree 
% 
% input
% -----
% A         nxn matrix
% c         constrained vector of length n. Components flagged with zero are
%           unconstrained. Positive values are assumed to be from 1,...,l and
%           move the constrained rows and columns to the end of the matrix
%           in the same order as in c
%
%
% output
% ------
% p         permutation vector. On exit A(p,p) refers to the reordered
%           system
%

if length(c)~=size(A,1)
   error('inconsistent constraints');
end

I=find(c);
if ~isempty(I)
   cc=sort(c(I));
   if cc(1)~=1 | cc(end)~=length(cc)
      error('inconsistent constraints');
   end
   [p]=symilupackcamd(0.5*(abs(A)+abs(A)'),c);
else
   [p]=symamd(A);
end
