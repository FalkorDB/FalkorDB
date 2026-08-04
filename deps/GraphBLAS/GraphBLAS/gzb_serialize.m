function C = gzb_serialize (ghb, A, method, level)
%GZB_SERIALIZE: wrapper for gbmex_serialize mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (ghb)
    switch (nargin)
        case 2
            C = GhB (gbmex_serialize (1, A)) ;
        case 3
            C = GhB (gbmex_serialize (1, A, method)) ;
        case 4
            C = GhB (gbmex_serialize (1, A, method, level)) ;
    end
else
    switch (nargin)
        case 2
            C = GrB (gbmex_serialize (0, A)) ;
        case 3
            C = GrB (gbmex_serialize (0, A, method)) ;
        case 4
            C = GrB (gbmex_serialize (0, A, method, level)) ;
    end
end

