function C = gb_isinf (ghb, G)
%GB_ISINF implements GrB/isinf and GhB/isinf.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

[m, n, type] = gbmex_size (G) ;

if (gb_isfloat (type))
    C = gzb_apply (ghb, 'isinf', G) ;
else
    % C is all false
    C = gzb (ghb, m, n, 'logical') ;
end

