function C = gb_isfinite (ghb, G)
%GB_ISFINITE implements isfinite for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

[m, n, type] = gbmex_size (G) ;

if (gb_isfloat (type) && m > 0 && n > 0)
    C = gzb_apply (ghb, 'isfinite', gzb_full (1, G)) ;
else
    % C is all true
    C = gb_true (ghb, m, n) ;
end

