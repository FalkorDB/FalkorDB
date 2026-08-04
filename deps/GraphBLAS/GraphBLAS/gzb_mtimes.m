function C = gzb_mtimes (ghb, A, B)
%GZB_MTIMES: wrapper for gbmex_mtimes mexFunction. Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

if (ghb)
    C = GhB (gbmex_mtimes (1, A, B)) ;
else
    C = GrB (gbmex_mtimes (0, A, B)) ;
end

