function d = gzb_degree (ghb, A, dim)
%GZB_DEGREE: wrapper for gbmex_degree mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (ghb)
    d = GhB (gbmex_degree (1, A, dim)) ;
else
    d = GrB (gbmex_degree (0, A, dim)) ;
end

