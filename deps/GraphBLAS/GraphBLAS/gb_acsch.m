function C = gb_acsch (ghb, G)
%GB_ACSCH implements GrB/acsch and GhB/acsch.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gzb_apply (ghb, 'asinh', gzb_apply (1, 'minv', gzb_full (1, G, type))) ;

