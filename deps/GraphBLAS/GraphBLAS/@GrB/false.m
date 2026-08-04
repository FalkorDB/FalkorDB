function C = false (varargin)
%GRB.FALSE a logical matrix with no entries.
%
%   C = GrB.false (n) ;      n-by-n GrB logical matrix with no entries.
%   C = GrB.false (m,n) ;    m-by-n GrB logical matrix with no entries.
%   C = GrB.false ([m,n]) ;  m-by-n GrB logical matrix with no entries.
%
% See also GrB.ones, GrB.true, GrB.zeros, GrB.eye, GrB.speye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_false (0, varargin {:}) ;

