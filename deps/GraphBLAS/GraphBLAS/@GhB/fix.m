function C = fix (G)
%FIX Round towards zero.
% C = fix (G) rounds the entries in the matrix G to the nearest integers
% towards zero.
%
% See also GhB/ceil, GhB/floor, GhB/round.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_fix (1, G) ;

