function X = gb_builtin (G)
%GB_BUILTIN wrapper for gbmex_builtin.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

X = gbmex_builtin (G) ;

