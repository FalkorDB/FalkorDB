function C = gzb_emult (ghb, arg1, arg2, arg3, desc)
%GZB_EMULT: wrapper for gbmex_emult mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

if (nargin < 5)
    desc = struct ;
end

if (ghb)
    C = GhB (gbmex_emult (1, arg1, arg2, arg3, desc)) ;
else
    C = GrB (gbmex_emult (0, arg1, arg2, arg3, desc)) ;
end

