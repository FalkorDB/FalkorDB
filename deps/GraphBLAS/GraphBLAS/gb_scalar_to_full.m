function C = gb_scalar_to_full (ghb, m, n, type, fmt, scalar)
%GB_SCALAR_TO_FULL expand a scalar into a full matrix.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (scalar))
    scalar = struct (scalar) ;
end

if (~isempty (strfind (fmt, 'by row'))) %#ok<STREMP>
    fmt = 'by row' ;
else
    fmt = 'by col' ;
end

E = gzb (1, m, n, type, fmt) ;
S = gzb_full (1, scalar) ;

if (ghb)
    C = GhB (gbmex_subassign (1, E, S)) ;
else
    C = GrB (gbmex_subassign (0, E, S)) ;
end

