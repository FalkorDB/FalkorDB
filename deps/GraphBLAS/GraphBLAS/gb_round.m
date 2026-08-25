function C = gb_round (ghb, G)
%GB_ROUND implements GrB/round and GhB/round.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: round (x,n) and round (x,n,type)

if (gb_is_grb (G))
    G = struct (G) ;
end

if (gb_isfloat (gbmex_type (G)))
    C = gzb_apply (ghb, 'round', G) ;
else
    C = gb_dup (ghb, G) ;
end

