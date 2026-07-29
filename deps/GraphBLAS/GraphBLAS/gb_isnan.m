function C = gb_isnan (ghb, G)
%GB_ISNAN implements GrB/isnan and GhB/isnan.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

[m, n, type] = gbmex_size (G) ;

if (gb_isfloat (type))
    C = gzb_apply (ghb, 'isnan', G) ;
else
    % C is all false
    C = gzb (ghb, m, n, 'logical') ;
end

