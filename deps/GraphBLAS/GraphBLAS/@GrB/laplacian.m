function L = laplacian (A, type, check)
%GRB.LAPLACIAN Laplacian matrix
% L = laplacian (A) is the graph Laplacian of the matrix A.  spones(A) must be
% symmetric.  The diagonal of A is ignored. The diagonal of L is the degree of
% the nodes.  That is, L(j,j) = sum (spones (A (:,j))), assuming A has no
% diagonal entries..  For off-diagonal entries, L(i,j) = L(j,i) = -1 if the
% edge (i,j) exists in A.
%
% The type of L defaults to double.  With a second argument, the type of L can
% be specified, as L = laplacian (A,type); type may be 'double', 'single',
% 'int8', 'int16', 'int32', 'int64', 'single complex', or 'double complex'.  Be
% aware that integer overflow may occur with the smaller integer types, if the
% degree of any nodes exceeds the largest integer value.
%
% spones(A) must be symmetric on input, but this condition is not checked by
% default.  If it is not symmetric, the results are undefined.  To check this
% condition, use GrB.laplacian (A, 'double', 'check') ;
%
% L is returned as symmetric GraphBLAS GrB matrix.
%
% Example:
%
%   A = bucky ;
%   L = GrB.laplacian (A)
%
% See also graph/laplacian.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

switch (nargin)
    case 1
        L = gb_laplacian (0, A) ;
    case 2
        L = gb_laplacian (0, A, type) ;
    case 3
        L = gb_laplacian (0, A, type, check) ;
end

