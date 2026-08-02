function C = conj (G)
%CONJ complex conjugate.
% C = conj (G) is the complex conjugate of each entry of G.
%
% See also GhB/real, GhB/imag.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_conj (1, G) ;

