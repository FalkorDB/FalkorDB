function C = zeros (varargin)
%GRB.ZEROS a matrix with no entries.
%
%   C = GrB.zeros (n) ;      n-by-n GrB double matrix with no entries.
%   C = GrB.zeros (m,n) ;    m-by-n GrB double matrix with no entries.
%   C = GrB.zeros ([m,n]) ;  m-by-n GrB double matrix with no entries.
%   C = GrB.zeros (..., type) ;      empty matrix of given type.
%   C = GrB.zeros (..., 'like', G) ; empty matrix, same type as G.
%
% See also GrB.ones, GrB.false, GrB.true, GrB.eye, GrB.speye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, type] = gb_parse_args ('zeros', varargin {:}) ;
C = GrB (gbnew (m, n, type)) ;

