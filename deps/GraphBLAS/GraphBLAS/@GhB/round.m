function C = round (G)
%ROUND round entries of a matrix to the nearest integers.
% C = round (G) rounds the entries of G to the nearest integers.
%
% Note: the additional parameters of the built-in round function, round(x,n)
% and round (x,n,type), are not supported.
%
% See also GhB/ceil, GhB/floor, GhB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_round (1, G) ;

