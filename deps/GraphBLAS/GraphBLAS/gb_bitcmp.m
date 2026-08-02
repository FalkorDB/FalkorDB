function C = gb_bitcmp (ghb, A, assumedtype)
%GB_BITCMP implements GrB/bitcmp and GhB/bitcmp.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (nargin < 3)
    assumedtype = 'uint64' ;
end

atype = gbmex_type (A) ;

if (gb_contains (atype, 'complex'))
    error ('GrB:error', 'inputs must be real') ;
end

if (isequal (atype, 'logical'))
    error ('GrB:error', 'inputs must not be logical') ;
end

if (~gb_contains (assumedtype, 'int'))
    error ('GrB:error', 'assumedtype must be an integer type') ;
end

% C will have the same type as A on input
ctype = atype ;

if (isequal (atype, 'double') || isequal (atype, 'single'))
    % cast A to the assumedtype
    T = gzb_full (1, gzb (1, A, assumedtype)) ;
else
    T = gzb_full (1, A) ;
end

C = gzb_apply (ghb, 'bitcmp', T) ;

if (~isequal (gb_type (C), ctype))
    C = gzb (ghb, C, ctype) ;
end

