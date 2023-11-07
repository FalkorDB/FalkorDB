function C = false (varargin)
%GRB.FALSE a logical matrix with no entries.
%
%   C = GrB.false (n) ;      n-by-n GrB logical matrix with no entries.
%   C = GrB.false (m,n) ;    m-by-n GrB logical matrix with no entries.
%   C = GrB.false ([m,n]) ;  m-by-n GrB logical matrix with no entries.
%   C = GrB.false (..., type) ;      empty logical matrix of given type.
%   C = GrB.false (..., 'like', G) ; empty logical matrix, same type as G.
%
% See also GrB.ones, GrB.true, GrB.zeros, GrB.eye, GrB.speye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, ~] = gb_parse_args ('false', varargin {:}) ;
C = GrB (gbnew (m, n, 'logical')) ;

