function C = gb_bitget (ghb, A, B, assumedtype)
%GB_BITGET implements GrB/bitget and GhB/bitget.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

if (nargin < 4)
    assumedtype = 'uint64' ;
end

atype = gbmex_type (A) ;
btype = gbmex_type (B) ;

if (gb_contains (atype, 'complex') || gb_contains (btype, 'complex'))
    error ('GrB:error', 'inputs must be real') ;
end

if (isequal (atype, 'logical') || isequal (btype, 'logical'))
    error ('GrB:error', 'inputs must not be logical') ;
end

if (~gb_contains (assumedtype, 'int'))
    error ('GrB:error', 'assumedtype must be an integer type') ;
end

% C will have the same type as A on input
ctype = atype ;

% ensure A and B have the right type then compute C = bitget (A,B)
if (isequal (atype, 'double') || isequal (atype, 'single'))
    atype = assumedtype ;
    op = ['bitget.' atype] ;
    A2 = gzb (1, A, atype) ;
    if (~isequal (btype, atype))
        C = gb_emult (ghb, A2, op, gzb (1, B, atype)) ;
    else
        C = gb_emult (ghb, A2, op, B) ;
    end
else
    op = ['bitget.' atype] ;
    if (~isequal (btype, atype))
        C = gb_emult (ghb, A, op, gzb (1, B, atype)) ;
    else
        C = gb_emult (ghb, A, op, B) ;
    end
end

if (~isequal (gb_type (C), ctype))
    C = gzb (ghb, C, ctype) ;
end

