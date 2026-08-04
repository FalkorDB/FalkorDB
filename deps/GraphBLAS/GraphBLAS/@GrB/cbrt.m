function C = cbrt (G)
%CBRT cube root
% C = cbrt (G) is the cube root of the entries of G.
%
% See also GrB/sqrt, nthroot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cbrt (0, G) ;

