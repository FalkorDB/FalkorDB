function C = gb_csc (ghb, G)
%GB_CSC implements GrB/csc and GhB/csc.  Not user-callable.

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

C = gzb_apply (ghb, 'minv', gzb_full (1, gzb_apply (1, op, G))) ;

