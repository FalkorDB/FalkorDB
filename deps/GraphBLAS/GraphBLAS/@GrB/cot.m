function C = cot (G)
%COT cotangent.
% C = cot (G) is the cotangent of each entry of G.  Since cot (0) is
% nonzero, C is a full matrix.
%
% See also GrB/coth, GrB/acot, GrB/acoth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cot (0, G) ;

