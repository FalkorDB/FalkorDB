function C = gzb_full (ghb, A, type, id, desc)
%GZB_FULL: wrapper for gbmex_full mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (nargin >= 4 && gb_is_grb (id))
    id = struct (id) ;
end

if (ghb)
    switch (nargin)
        case 2
            C = GhB (gbmex_full (1, A)) ;
        case 3
            C = GhB (gbmex_full (1, A, type)) ;
        case 4
            C = GhB (gbmex_full (1, A, type, id)) ;
        case 5
            C = GhB (gbmex_full (1, A, type, id, desc)) ;
    end
else
    switch (nargin)
        case 2
            C = GrB (gbmex_full (0, A)) ;
        case 3
            C = GrB (gbmex_full (0, A, type)) ;
        case 4
            C = GrB (gbmex_full (0, A, type, id)) ;
        case 5
            C = GrB (gbmex_full (0, A, type, id, desc)) ;
    end
end

