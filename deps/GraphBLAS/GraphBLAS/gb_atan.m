function C = gb_atan (ghb, G)
%GB_ATAN implements GrB/atan and GhB/atan.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (~gb_isfloat (gbmex_type (G)))
    op = 'atan.double' ;
else
    op = 'atan' ;
end

C = gzb_apply (ghb, op, G) ;

