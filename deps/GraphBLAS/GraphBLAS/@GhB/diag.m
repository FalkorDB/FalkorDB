function C = diag (A, k)
%DIAG diagonal matrices and diagonals of a matrix.
% C = diag (v,k) when v is a vector with n components is a square sparse matrix
% of dimension n+abs(k), with the elements of v on the kth diagonal. The main
% diagonal is k = 0; k > 0 is above the diagonal, and k < 0 is below the main
% diagonal.  C = diag (v) is the same as C = diag (v,0).
%
% c = diag (A,k) when A is a matrix returns a column vector c formed the
% entries on the kth diagonal of A.  The main diagonal is c = diag (A).
%
% The GraphBLAS diag function always constructs a GraphBLAS sparse matrix,
% unlike the built-in diag, which always constructs a full matrix.  To use this
% overloaded function for a non-GhB sparse matrix A, use C = diag (A, GhB (k)).
%
% Examples:
%
%   C1 = diag (GhB (1:10, 'uint8'), 2)
%   C2 = sparse (diag (1:10, 2))
%   nothing = double (C1-C2)
%
%   A = magic (8)
%   full (double ([diag(A,1) diag(GhB(A),1)]))
%
%   m = 5 ;
%   f = ones (2*m,1) ;
%   A = diag(-m:m) + diag(f,1) + diag(f,-1)
%   G = diag(GhB(-m:m)) + diag(GhB(f),1) + diag(GhB(f),-1)
%   nothing = double (A-G)
%
% See also GhB/diag, spdiags, GhB/tril, GhB/triu, GhB.select.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    C = gb_diag (1, A) ;
else
    C = gb_diag (1, A, k) ;
end

