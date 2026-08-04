function C = true (varargin)
%GHB.TRUE a logical matrix with all true values.
%
%   C = GhB.true (n) ;      n-by-n GhB logical matrix of all true entries.
%   C = GhB.true (m,n) ;    m-by-n GhB logical matrix of all true entries.
%   C = GhB.true ([m,n]) ;  m-by-n GhB logical matrix of all true entries.
%
% The memory required to store C is O(1) not O(m*n), so both m and n can be as
% large as 2^60.
%
% See also GhB.zeros, GhB.ones, GhB.false, GhB.eye, GhB.speye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_true (1, varargin {:}) ;

