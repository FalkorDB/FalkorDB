function C = gzb_select (ghb, arg1, arg2, arg3, arg4)
%GZB_SELECT: wrapper for gbmex_select mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (nargin >= 4 && gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

% if arg4 is present, it must be the descriptor

if (ghb)
    switch (nargin)
        case 3
            C = GhB (gbmex_select (1, arg1, arg2)) ;
        case 4
            C = GhB (gbmex_select (1, arg1, arg2, arg3)) ;
        case 5
            C = GhB (gbmex_select (1, arg1, arg2, arg3, arg4)) ;
    end
else
    switch (nargin)
        case 3
            C = GrB (gbmex_select (0, arg1, arg2)) ;
        case 4
            C = GrB (gbmex_select (0, arg1, arg2, arg3)) ;
        case 5
            C = GrB (gbmex_select (0, arg1, arg2, arg3, arg4)) ;
    end
end

