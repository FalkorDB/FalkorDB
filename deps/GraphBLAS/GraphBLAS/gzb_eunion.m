function C = gzb_eunion (ghb, A, op, B)
%GZB_EUNION: wrapper for gbmex_eunion mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

alpha = 0 ;
beta = 0 ;

if (ghb)
    C = GhB (gbmex_eunion (1, A, alpha, op, B, beta)) ;
else
    C = GrB (gbmex_eunion (0, A, alpha, op, B, beta)) ;
end

