function C = gb_sin (ghb, G)
%GB_SIN implements GrB/sin and GhB/sin.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (~gb_isfloat (gbmex_type (G)))
    op = 'sin.double' ;
else
    op = 'sin' ;
end

C = gzb_apply (ghb, op, G) ;

