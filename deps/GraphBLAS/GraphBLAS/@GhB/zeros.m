function C = zeros (varargin)
%GHB.ZEROS a matrix with no entries.
%
%   C = GhB.zeros (n) ;      n-by-n GhB double matrix with no entries.
%   C = GhB.zeros (m,n) ;    m-by-n GhB double matrix with no entries.
%   C = GhB.zeros ([m,n]) ;  m-by-n GhB double matrix with no entries.
%   C = GhB.zeros (..., type) ;      empty matrix of given type.
%   C = GhB.zeros (..., 'like', G) ; empty matrix, same type as G.
%
% See also GhB.ones, GhB.false, GhB.true, GhB.eye, GhB.speye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_zeros (1, varargin {:}) ;

