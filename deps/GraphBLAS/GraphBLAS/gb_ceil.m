function C = gb_ceil (ghb, G)
%GB_CEIL implements GrB/ceil and GhB/ceil.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (gb_isfloat (gbmex_type (G)))
    C = gzb_apply (ghb, 'ceil', G) ;
else
    C = gb_dup (ghb, G) ;
end

