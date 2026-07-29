function C = gb_cast_full (G, type, zero)
%GB_CAST_FULL cast to a full matrix.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

desc.kind = 'full' ;
C = gb_builtin (gzb_full (1, G, type, zero, desc)) ;

