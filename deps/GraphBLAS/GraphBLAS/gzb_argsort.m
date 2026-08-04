function [C,P] = gzb_argsort (ghb, A, dim, direction)
%GZB_ARGSORT: wrapper for gbmex_argsort mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (ghb)
    if (nargout == 1)
        C = GhB (gbmex_argsort (1, A, dim, direction)) ;
    else
        [C_opaque, P_opaque] = gbmex_argsort (1, A, dim, direction) ;
        C = GhB (C_opaque) ;
        P = GhB (P_opaque) ;
    end
else
    if (nargout == 1)
        C = GrB (gbmex_argsort (0, A, dim, direction)) ;
    else
        [C_opaque, P_opaque] = gbmex_argsort (0, A, dim, direction) ;
        C = GrB (C_opaque) ;
        P = GrB (P_opaque) ;
    end
end

