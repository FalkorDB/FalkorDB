function [F, E] = log2 (G)
%LOG2 base-2 logarithm.
% C = log2 (G) is the base-2 logarithm of each entry of a GraphBLAS matrix G.
% Since log2 (0) is nonzero, the result is a full matrix.  If any entry in G is
% negative, the result is complex.
%
% [F,E] = log2 (G) returns F and E so that G = F.*(2.^E), where entries in abs
% (F) are either in the range [0.5,1), or zero if the entry in G is zero.  F
% and E are both sparse, with the same pattern as G.  If G is complex,
% [F,E] = log2 (real (G)).
%
% See also GhB/pow2, GhB/log, GhB/log1p, GhB/log10, GhB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargout == 1)
    F = gb_log2 (1, G) ;
else
    [F, E] = gb_log2 (1, G) ;
end

