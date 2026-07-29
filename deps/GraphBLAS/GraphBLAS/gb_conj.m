function C = gb_conj (ghb, G)
%GB_CONJ implements GrB/conj and GhB/conj.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (gb_contains (gbmex_type (G), 'complex'))
    C = gzb_apply (ghb, 'conj', G) ;
else
    C = gb_dup (ghb, G) ;
end

