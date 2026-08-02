function C = gzb_vreduce (ghb, arg1, arg2, arg3, arg4, desc)
%GZB_VREDUCE: wrapper for gbmex_vreduce mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% matrices are provided as the 1 to 3 inputs to this method

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (nargin >= 4 && gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

if (ghb)
    switch (nargin)
        case 3
            C = GhB (gbmex_vreduce (1, arg1, arg2)) ;
        case 4
            C = GhB (gbmex_vreduce (1, arg1, arg2, arg3)) ;
        case 5
            C = GhB (gbmex_vreduce (1, arg1, arg2, arg3, arg4)) ;
        case 6
            C = GhB (gbmex_vreduce (1, arg1, arg2, arg3, arg4, desc)) ;
    end
else
    switch (nargin)
        case 3
            C = GrB (gbmex_vreduce (0, arg1, arg2)) ;
        case 4
            C = GrB (gbmex_vreduce (0, arg1, arg2, arg3)) ;
        case 5
            C = GrB (gbmex_vreduce (0, arg1, arg2, arg3, arg4)) ;
        case 6
            C = GrB (gbmex_vreduce (0, arg1, arg2, arg3, arg4, desc)) ;
    end
end

