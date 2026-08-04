function C = ones (varargin)
%GHB.ONES a matrix with all ones.
%
%   C = GhB.ones (n) ;      n-by-n GhB double matrix of all ones.
%   C = GhB.ones (m,n) ;    m-by-n GhB double matrix of all ones.
%   C = GhB.ones ([m,n]) ;  m-by-n GhB double matrix of all ones.
%   C = GhB.ones (..., type) ;      matrix of all ones of given type.
%   C = GhB.ones (..., 'like', G) ; matrix of all ones, same type as G.
%
% The memory required to store C is O(1) not O(m*n), so both m and n can be as
% large as 2^60.
%
% See also GhB.zeros, GhB.false, GhB.true, GhB.eye, GhB.speye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_ones (1, varargin {:}) ;

