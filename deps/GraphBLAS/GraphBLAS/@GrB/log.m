function C = log (G)
%LOG natural logarithm.
% C = log (G) is the natural logarithm of each entry of G.
% Since log (0) is nonzero, the result is a full matrix.
% If any entry in G is negative, the result is complex.
%
% See also GrB/log1p, GrB/log2, GrB/log10, GrB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
C = GrB (gb_make_real (gb_trig ('log', gbfull (G)))) ;

