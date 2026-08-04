function C = log (G)
%LOG natural logarithm.
% C = log (G) is the natural logarithm of each entry of G.  Since log (0) is
% nonzero, the result is a full matrix.  If any entry in G is negative, the
% result is complex.
%
% See also GhB/log1p, GhB/log2, GhB/log10, GhB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_log (1, G) ;

