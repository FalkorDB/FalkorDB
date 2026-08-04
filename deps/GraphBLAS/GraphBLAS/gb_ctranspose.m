function C = gb_ctranspose (ghb, G)
%GB_CTRANSPOSE implements G' for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (gb_contains (gbmex_type (G), 'complex'))
    desc.in0 = 'transpose' ;
    C = gzb_apply (ghb, 'conj', G, desc) ;
else
    C = gzb_trans (ghb, G) ;
end

