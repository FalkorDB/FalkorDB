function C = gb_tan (ghb, G)
%GB_TAN implements GrB/tan and GhB/tan.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (~gb_isfloat (gbmex_type (G)))
    op = 'tan.double' ;
else
    op = 'tan' ;
end

C = gzb_apply (ghb, op, G) ;

