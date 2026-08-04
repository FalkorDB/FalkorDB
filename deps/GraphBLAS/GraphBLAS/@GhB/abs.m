function C = abs (G)
%ABS absolute value.
% C = abs (G) is the absolute value of each entry of G.
%
% See also GhB/sign.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_abs (1, G) ;

