function C = gb_asec (ghb, G)
%GB_ASEC implements GrB/asec and GhB/asec.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gb_trig (ghb, 'acos', gzb_apply (1, 'minv', gzb_full (1, G, type))) ;

