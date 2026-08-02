function C = gb_asech (ghb, G)
%GB_ASECH implements GrB/asech and GhB/asech.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gb_trig (ghb, 'acosh', gzb_apply (1, 'minv', gzb_full (1, G, type))) ;

