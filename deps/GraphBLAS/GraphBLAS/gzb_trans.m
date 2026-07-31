function C = gzb_trans (ghb, A)
%GZB_TRANS: wrapper for gbmex_trans mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (ghb)
    C = GhB (gbmex_trans (1, A)) ;
else
    C = GrB (gbmex_trans (0, A)) ;
end

