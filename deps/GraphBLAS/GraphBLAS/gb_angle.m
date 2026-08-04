function C = gb_angle (ghb, G)
%GB_ANGLE implements GrB/angle and GhB/angle.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

[m, n, type] = gbmex_size (G) ;
if (gb_contains (type, 'complex'))
    C = gzb_apply (ghb, 'carg', G) ;
else
    % C is all zero
    C = gzb (ghb, m, n, type) ;
end

