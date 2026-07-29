function C = gb_expm1 (ghb, G)
%GB_EXPM1 implements GrB/expm1 and GhB/expm1.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (~gb_isfloat (gbmex_type (G)))
    op = 'expm1.double' ;
else
    op = 'expm1' ;
end

C = gzb_apply (ghb, op, G) ;

