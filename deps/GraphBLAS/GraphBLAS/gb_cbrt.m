function C = gb_cbrt (ghb, G)
%GB_CBRT implements GrB/cbrt and GhB/cbrt.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

type = gbmex_type (G) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'input must be real') ;
elseif (gb_isfloat (type))
    op = 'cbrt' ;
else
    op = 'cbrt.double' ;
end

C = gzb_apply (ghb, op, G) ;

