function C = false (varargin)
%GHB.FALSE a logical matrix with no entries.
%
%   C = GhB.false (n) ;      n-by-n GhB logical matrix with no entries.
%   C = GhB.false (m,n) ;    m-by-n GhB logical matrix with no entries.
%   C = GhB.false ([m,n]) ;  m-by-n GhB logical matrix with no entries.
%
% See also GhB.ones, GhB.true, GhB.zeros, GhB.eye, GhB.speye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_false (1, varargin {:}) ;

