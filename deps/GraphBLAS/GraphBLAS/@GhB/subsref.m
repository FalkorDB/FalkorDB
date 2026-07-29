function C = subsref (A, S)
%SUBSREF C = A(I,J) or C = A(I); extract submatrix.
% C = A(I,J) extracts the A(I,J) submatrix of the GraphBLAS matrix A.  With a
% single index, C = A(I) extracts a subvector C of a vector A.  For linear
% indexing of a 2D matrix, only C=A(:) is currently supported.  C = A(I) is not
% yet supported if A is a 2D matrix.
%
% x = A (M) for a logical matrix M constructs an nnz(M)-by-1 vector x, for
% built-in-style logical indexing.  A or M may be built-in sparse or full
% matrices, or GraphBLAS matrices, in any combination.  M must be either a
% built-in logical matrix (sparse or full), or a GraphBLAS logical matrix; that
% is, GhB.type (M) must be 'logical'.
%
% GraphBLAS can construct huge sparse matrices, but they cannot always be
% indexed with A(lo:hi,lo:hi), because of a limitation of the built-in colon
% notation.  A colon expression is expanded into an explicit vector, but this
% can be too big.   Instead of the colon notation start:inc:fini, use a cell
% array with three integers, {start, inc, fini}.
%
% Example:
%
%   n = 1e14 ;
%   H = GhB (n, n)               % a huge empty matrix
%   I = [1 1e9 1e12 1e14] ;
%   M = magic (4)
%   H (I,I) = M
%   J = {1, 1e13} ;             % represents 1:1e13 colon notation
%   C = H (J, J)                % this is very fast
%   E = H (1:1e13, 1:1e13)      % but this is not possible
%
% See also GhB/subsasgn, GhB/subsindex, GhB.subassign, GhB.assign,
% GhB.extract.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_subsref (1, A, S) ;

