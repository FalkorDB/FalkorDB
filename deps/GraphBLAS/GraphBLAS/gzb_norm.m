function s = gzb_norm (G, kind)
%GZB_NORM wrapper for gbmex_norm.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

s = gbmex_norm (G, kind) ;

