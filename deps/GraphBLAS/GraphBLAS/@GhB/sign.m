function C = sign (G)
%SIGN signum function.
% C = sign (G) is the signum function for each entry of G.  For real values,
% sign(x) is 1 if x > 0, zero if x is zero, and -1 if x < 0.  For the complex
% case, sign(x) = x ./ abs (x).
%
% See also GhB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_sign (1, G) ;

