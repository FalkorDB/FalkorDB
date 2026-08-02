function C = gb_sinh (ghb, G)
%GB_SINH implements GrB/sinh and GhB/sinh.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (~gb_isfloat (gbmex_type (G)))
    op = 'sinh.double' ;
else
    op = 'sinh' ;
end

C = gzb_apply (ghb, op, G) ;

