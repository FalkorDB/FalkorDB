function C = gb_tanh (ghb, G)
%GB_TANH implements GrB/tanh and GhB/tanh.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (~gb_isfloat (gbmex_type (G)))
    op = 'tanh.double' ;
else
    op = 'tanh' ;
end

C = gzb_apply (ghb, op, G) ;

