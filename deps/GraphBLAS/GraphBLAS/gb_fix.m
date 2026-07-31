function C = gb_fix (ghb, G)
%GB_FIX implements GrB/fix and GhB/fix.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (gb_isfloat (gbmex_type (G)))
    C = gzb_apply (ghb, 'trunc', G) ;
else
    C = gb_dup (ghb, G) ;
end

