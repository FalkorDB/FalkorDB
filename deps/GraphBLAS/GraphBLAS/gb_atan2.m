function C = gb_atan2 (ghb, A, B)
%GB_ATAN2: implements GrB/atan2 and GhB/atan2.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

atype = gbmex_type (A) ;
btype = gbmex_type (B) ;

if (gb_contains (atype, 'complex') || gb_contains (btype, 'complex'))
    error ('GrB:error', 'inputs must be real') ;
end

% cast A and/or B to double, if not already a floating-point type
if (gb_isfloat (atype))
    if (gb_isfloat (btype))
        C = gb_atan2b (ghb, A, B) ;
    else
        C = gb_atan2b (ghb, A, gzb (1, B, 'double')) ;
    end
else
    if (gb_isfloat (btype))
        C = gb_atan2b (ghb, gzb (1, A, 'double'), B) ;
    else
        C = gb_atan2b (ghb, gzb (1, A, 'double'), gzb (1, B, 'double')) ;
    end
end

