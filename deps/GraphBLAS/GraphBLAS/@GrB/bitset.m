function C = bitset (A, B, arg3, arg4)
%BITSET set bit.
% C = bitset (A,B) sets a bit in A to 1, where the bit position is
% determined by B.  A is an integer array.  If B(i,j) is an integer in the
% range 1 (the least significant bit) to the number of bits in the data
% type of A, then C(i,j) is equal to the value of A(i,j) after setting the
% bit to 1.  If B(i,j) is outside this range, C(i,j) is set to A(i,j),
% unmodified; note that this behavior is an extension of the built-in
% bitset, which results in an error for this case.  This modified
% rule allows the inputs A and B to be sparse.
%
% If A and B are matrices, the pattern of C is the set union of A
% and B.  If one of A or B is a nonzero scalar, the scalar is expanded
% into a sparse matrix with the same pattern as the other matrix, and the
% result is a sparse matrix.
%
% If the last input argument is a string, C = bigset (A,B,assumedtype)
% provides a data type to convert A to if it has a floating-point type.
% If A already has an integer type, then it is not modified.  Otherwise, A
% is converted to assumedtype, which can be 'int8', 'int16', 'int32',
% 'int64', 'uint8', 'uint16', 'uint32' or 'uint64'.  The default is
% 'uint64'.
%
% C = bitset (A,B,V) sets the bit in A(i,j) at position B(i,j) to 0 if
% V(i,j) is zero, or to 1 if V(i,j) is nonzero.  If V is a scalar, it
% is implicitly expanded to V * spones (B).
%
% All four arguments may be used, as C = bitset (A,B,V,assumedtype).
%
% Example:
%
%   A = GrB (magic (4), 'uint8')
%   B = reshape ([1:8 1:8], 4, 4)
%   C = bitset (A, B)
%   fprintf ('\nA: ') ; fprintf ('%3x ', A) ; fprintf ('\n') ;
%   fprintf ('\nB: ') ; fprintf ('%3x ', B) ; fprintf ('\n') ;
%   fprintf ('\nC: ') ; fprintf ('%3x ', C) ; fprintf ('\n') ;
%   C2 = bitset (uint8 (A), B)
%   isequal (C2, C)
%
% See also GrB/bitor, GrB/bitand, GrB/bitxor, GrB/bitcmp, GrB/bitshift,
% GrB/bitset, GrB/bitclr.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

[am, an, atype] = gbsize (A) ;
[bm, bn, btype] = gbsize (B) ;

if (gb_contains (atype, 'complex') || gb_contains (btype, 'complex'))
    error ('GrB:error', 'inputs must be real') ;
end

if (isequal (atype, 'logical') || isequal (btype, 'logical'))
    error ('GrB:error', 'inputs must not be logical') ;
end

a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;

% get the optional input arguments
if (nargin == 4)
    V = arg3 ;
    assumedtype = arg4 ;
elseif (nargin == 3)
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
if (isequal (atype, 'double') || isequal (atype, 'single'))
    A = gbnew (A, assumedtype) ;
    atype = assumedtype ;
end

% ensure B has the same type as A
if (~isequal (btype, atype))
    B = gbnew (B, atype) ;
end

% get the matrix or scalar V
if (isobject (V))
    V = V.opaque ;
end
[m, n] = gbsize (V) ;
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
            C = gbeunion (op, A, 0, B, 0) ;
        else
            % A is a scalar, B is a matrix
            C = gbapply2 (op, gbfull (A), B) ;
        end
    else
        % A is a matrix
        if (b_is_scalar)
            % A is a matrix, B is scalar
            C = gbapply2 (op, A, gbfull (B)) ;
        else
            % both A and B are matrices
            C = gbeunion (op, A, 0, B, 0) ;
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
        A = gb_scalar_to_full (m, n, atype, gb_fmt (V), A) ;
    end
    if (b_is_scalar)
        % expand B to a full matrix the same size as V.
        B = gb_scalar_to_full (m, n, atype, gb_fmt (V), B) ;
    end

    % Set all bits referenced by B(i,j) to 1, even those that need to be
    % set to 0, without considering V(i,j).
    C = gbeunion (['bitset.', atype], A, 0, B, 0) ;

    % The pattern of C is now the set intersection of A and B, but
    % bits referenced by B(i,j) have been set to 1, not 0.  Construct B0
    % as the bits in B(i,j) that must be set to 0; B0<~V>=B defines the
    % pattern of bit positions B0 to set to 0 in A.
    d.mask = 'complement' ;
    B0 = gbassign (gbnew (m, n, atype), V, B, d) ;

    % Clear the bits in C, referenced by B0(i,j), where V(i,j) is zero.
    C = gbeadd (['bitclr.', atype], C, B0) ;

end

% return result
if (isequal (gbtype (C), ctype))
    C = GrB (C) ;
else
    C = GrB (gbnew (C, ctype)) ;
end


