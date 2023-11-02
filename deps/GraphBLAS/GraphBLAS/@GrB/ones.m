function C = ones (varargin)
%GRB.ONES a matrix with all ones.
%
%   C = GrB.ones (n) ;      n-by-n GrB double matrix of all ones.
%   C = GrB.ones (m,n) ;    m-by-n GrB double matrix of all ones.
%   C = GrB.ones ([m,n]) ;  m-by-n GrB double matrix of all ones.
%   C = GrB.ones (..., type) ;      matrix of all ones of given type.
%   C = GrB.ones (..., 'like', G) ; matrix of all ones, same type as G.
%
% The memory required to store C is O(1) not O(m*n), so both m and
% n can be as large as 2^60.
%
% See also GrB.zeros, GrB.false, GrB.true, GrB.eye, GrB.speye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, type] = gb_parse_args ('ones', varargin {:}) ;
C = GrB (gb_scalar_to_full (m, n, type, gbformat, 1)) ;

