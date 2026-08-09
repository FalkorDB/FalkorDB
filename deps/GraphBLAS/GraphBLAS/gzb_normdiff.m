function s = gzb_normdiff (A, B, kind)
%GZB_NORMDIFF wrapper for gbmex_normdiff.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

s = gbmex_normdiff (A, B, kind) ;

