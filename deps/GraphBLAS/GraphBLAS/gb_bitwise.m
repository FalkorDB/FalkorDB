function C = gb_bitwise (ghb, op, A_arg, B_arg, assumedtype)
%GB_BITWISE bitwise AND, OR, XOR, ...  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A_arg))
    A_arg = struct (A_arg) ;
end

if (gb_is_grb (B_arg))
    B_arg = struct (B_arg) ;
end

if (nargin < 5)
    assumedtype = 'uint64' ;
end

atype = gbmex_type (A_arg) ;
btype = gbmex_type (B_arg) ;

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

if (isequal (atype, 'double') || isequal (atype, 'single'))
    A = gzb (1, A_arg, assumedtype) ;
    atype = assumedtype ;
else
    A = A_arg ;
end

if (isequal (op, 'bitshift'))

    if (~isequal (btype, 'int8'))
        % convert B to int8, and ensure all values are in range -64:64
        % ensure all entries in B are <= 64
        B = gzb_apply2 (1, ['min.' btype], B_arg, 64) ;
        if (gb_issigned (btype))
            % ensure all entries in B are >= -64
            B = gzb_apply2 (1, ['max.' btype], B, -64) ;
        end
        B = gzb (1, B, 'int8') ;
    else
        B = B_arg ;
    end

    a_is_scalar = gb_isscalar (A) ;
    b_is_scalar = gb_isscalar (B) ;

    if (a_is_scalar && ~b_is_scalar)
        % A is a scalar, B is a matrix
        C = gzb_apply2 (ghb, ['bitshift.' atype], gzb_full (1, A), B) ;
    elseif (~a_is_scalar && b_is_scalar)
        % A is a matrix, B is a scalar
        C = gzb_apply2 (ghb, ['bitshift.' atype], A, gzb_full (1, B)) ;
    else
        % both A and B are matrices, or both are scalars
        % expand B by padding it with zeros from the pattern of A
        b = gzb_eadd (1, '1st.int8', B, gb_expand (1, 0, A, 'int8')) ;
        C = gzb_emult (ghb, ['bitshift.' atype], A, b) ;
    end

else

    if (isequal (btype, 'double') || isequal (btype, 'single'))
        B = gzb (1, B_arg, assumedtype) ;
        btype = assumedtype ;
    else
        B = B_arg ;
    end
    if (~isequal (atype, btype))
        error ('GrB:error', 'integer inputs must have the same type') ;
    end

    switch (op)
        case { 'bitxor', 'bitor' }
            C = gb_eadd (ghb, A, op, B) ;
        case { 'bitand' }
            C = gb_emult (ghb, A, op, B) ;
    end
end

if (~isequal (gb_type (C), ctype))
    C = gzb (ghb, C, ctype) ;
end

