function C = gb_bitset (ghb, A_arg, B_arg, arg3, arg4)
%GB_BITSET: implements GrB.bitset and GhB.bitset.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A_arg))
    A_arg = struct (A_arg) ;
end

if (gb_is_grb (B_arg))
    B_arg = struct (B_arg) ;
end

if (nargin >= 4 && gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

[am, an, atype] = gbmex_size (A_arg) ;
[bm, bn, btype] = gbmex_size (B_arg) ;

if (gb_contains (atype, 'complex') || gb_contains (btype, 'complex'))
    error ('GrB:error', 'inputs must be real') ;
end

if (isequal (atype, 'logical') || isequal (btype, 'logical'))
    error ('GrB:error', 'inputs must not be logical') ;
end

a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;

% get the optional input arguments
if (nargin == 5)
    V = arg3 ;
    assumedtype = arg4 ;
elseif (nargin == 4)
    if (ischar (arg3))
        V = 1 ;
        assumedtype = arg3 ;
    else
        V = arg3 ;
        assumedtype = 'uint64' ;
    end
else
    V = 1 ;
    assumedtype = 'uint64' ;
end

if (~gb_contains (assumedtype, 'int'))
    error ('GrB:error', 'assumedtype must be an integer type') ;
end

% C will have the same type as A on input
ctype = atype ;

% determine the type of A
cast_A = isequal (atype, 'double') || isequal (atype, 'single') ;
if (cast_A)
    A = gzb (1, A_arg, assumedtype) ;
    atype = assumedtype ;
else
    % use the input A_arg as-is
    A = A_arg ;
end

% ensure B has the same type as A
cast_B = ~isequal (btype, atype) ;
if (cast_B)
    B = gzb (1, B_arg, atype) ;
else
    % use the input B_arg as-is
    B = B_arg ;
end

% get the matrix or scalar V
[m, n] = gbmex_size (V) ;
V_is_scalar = (m == 1) && (n == 1) ;

if (V_is_scalar)

    % V is a scalar:  all bits in A indexed by B are either cleared or set.
    if (gb_scalar (V) == 0)
        % any bit reference by B(i,j) is set to 0 in A
        op = ['bitclr.' atype] ;
    else
        % any bit reference by B(i,j) is set to 1 in A
        op = ['bitset.' atype] ;
    end

    if (a_is_scalar)
        % A is a scalar
        if (b_is_scalar)
            % both A and B are scalars
            C = gzb_eunion (ghb, A, op, B) ;
        else
            % A is a scalar, B is a matrix
            a = gzb_full (1, A) ;
            C = gzb_apply2 (ghb, op, a, B) ;
        end
    else
        % A is a matrix
        if (b_is_scalar)
            % A is a matrix, B is scalar
            b = gzb_full (1, B) ;
            C = gzb_apply2 (ghb, op, A, b) ;
        else
            % both A and B are matrices
            C = gzb_eunion (ghb, A, op, B) ;
        end
    end

else

    % V is a matrix: A and B can be scalars or matrices, but if they
    % are matrices, they must have the same size as V.

    % if B(i,j) is nonzero and V(i,j)=1, then:
    % C(i,j) = bitset (A (i,j), B (i,j)).

    % if B(i,j) is nonzero and V(i,j)=0 (implicit or explicit), then:
    % C(i,j) = bitclr (A (i,j), B (i,j)).

    if (a_is_scalar)
        % expand A to a full matrix the same size as V.
        A2 = gb_scalar_to_full (1, m, n, atype, gb_fmt (V), A) ;
    else
        A2 = A ;
    end
    if (b_is_scalar)
        % expand B to a full matrix the same size as V.
        B2 = gb_scalar_to_full (1, m, n, atype, gb_fmt (V), B) ;
    else
        B2 = B ;
    end

    % Set all bits referenced by B(i,j) to 1, even those that need to be
    % set to 0, without considering V(i,j).
    S = gzb_eunion (1, A2, ['bitset.', atype], B2) ;

    % The pattern of S is now the set intersection of A and B, but
    % bits referenced by B(i,j) have been set to 1, not 0.  Construct B0
    % as the bits in B(i,j) that must be set to 0; B0<~V>=B defines the
    % pattern of bit positions B0 to set to 0 in A.
    desc.mask = 'complement' ;
    E = gzb (1, m, n, atype) ;
    B0 = GhB (gbmex_subassign (1, E, V, B2, desc)) ;

    % Clear the bits in C, referenced by B0(i,j), where V(i,j) is zero.
    C = gzb_eadd (ghb, ['bitclr.', atype], S, B0) ;

end

% return result
if (~isequal (gb_type (C), ctype))
    C = gzb (ghb, C, ctype) ;
end

