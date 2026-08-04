function C = exp (G)
%EXP exponential.
% C = exp (G) is e^x for each entry x of the matrix G.
% Since e^0 is nonzero, C is a full matrix.
%
% See also GhB/exp, GhB/expm1, GhB/pow2, GhB/log, GhB/log10, GhB/log2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_exp (1, G) ;

