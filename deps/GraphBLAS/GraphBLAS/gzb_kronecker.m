function C = gzb_kronecker (ghb, A, op, B)
%GZB_KRONECKER: wrapper for gbmex_kronecker mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

if (ghb)
    C = GhB (gbmex_kronecker (1, A, op, B)) ;
else
    C = GrB (gbmex_kronecker (0, A, op, B)) ;
end

