function C = gb_real (ghb, G)
%GB_REAL implements GrB/real and GhB/real.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (gb_contains (gbmex_type (G), 'complex'))
    C = gzb_apply (ghb, 'creal', G) ;
else
    % G is already real
    C = gb_dup (ghb, G) ;
end

