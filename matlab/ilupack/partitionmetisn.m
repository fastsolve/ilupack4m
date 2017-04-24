function [p,rangtab,treetab] = partitionmetisn(A,nleaves)
% [p,istart,parent] = partitionmetisn(A,nleaves)
% 
% partition a given nxn matrix A using METIS multilevel nested dissection
% by nodes
% 
% input
% -----
% A         n x n  matrix
% nleaves   number of partitionings, only powers of 2 are admissible. If
%           violated, the nearest power of 2 is chosen
%
% output
% ------
% p         permutation vector. On exit A(p,p) refers to the reordered
%           system
%           The parallel partitioning is based on an (incomplete) binary tree
%           with 2*nleaves-1 nodes.
%           It is set up in a hierarchical fashion such that parent nodes
%           follow their children.
%
% istart    describes the starting index each node in the binary tree
%           associated with the reordered system A(p,p). Consistently,
%           range(2*nleaves)=n+1
%
% parent    returns the parent information for each node. There are 2*nleaves-1
%           nodes in a binary tree. '0' refers to the root node, which does not
%           have a parent

[p,rangtab,treetab]=partitionilupackmetisn(A,nleaves);
