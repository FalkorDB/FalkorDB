function C = gb_erf (ghb, G)
%GB_ERF implements GrB/erf and GhB/erf.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

type = gbmex_type (G) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'input must be real') ;
end
if (~gb_isfloat (type))
    op = 'erf.double' ;
else
    op = 'erf' ;
end

C = gzb_apply (ghb, op, G) ;

