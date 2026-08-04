function C = flip (A, dim)
%FLIP flip the order of elements.
% C = flip (A) flips the order of elements in each column of A.  That is,
% C = A (end:-1:1,:).  C = flip (A, dim) specifies the dimension to flip, so
% that flip (A,1) and flip (A) are the same thing, and flip (A,2) flips the
% columns so that C = A (:,end:-1,1).
%
% To use this function on a built-in matrix, use C = flip (A, GrB (dim)).
%
% See also GrB/transpose.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    C = gb_flip (0, A) ;
else
    C = gb_flip (0, A, dim) ;
end

