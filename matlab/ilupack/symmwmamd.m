function [p,D] = symmwmamd(A,ind)
% [p,D] = symmwmamd(A)
% [p,D] = symmwmamd(A,ind)
% 
% reorder and rescale a given nxn SYMMETRIC/HERMITIAN matrix A using symmetric
% maximum weight matching followed by approximate minimum degree
% 
% input
% -----
% A         nxn matrix
% ind       optionally, vector of size n (size of A), where negative entries
%           indicate a second block in a block-structured A such as
%           [A B; B' 0] (Stokes-type problem). The block structure could be
%           up to permutation.
%
% output
% ------
% p         permutation vector. On exit D*A(p,p)*D refers to the reordered
%           and rescaled system
%

if nargin==1
   [p,D]=symmwmilupackamd(0.5*(abs(A)+abs(A)'));
else
   [p,D]=symmwmilupackamdsp(0.5*(abs(A)+abs(A)'),ind);
end
n=size(A,1);
D=spdiags(D(p),0,n,n);
