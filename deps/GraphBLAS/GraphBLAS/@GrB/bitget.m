function C = bitget (A, B, assumedtype)
%BITGET get bit.
% C = bitget (A,B) returns the value of the bit at position B in A, where
% A is an integer array.  If B(i,j) is an integer in the range 1 (the
% least significant bit) to the number of bits in the data type of A, then
% C(i,j) is that bit of A(i,j).  If B(i,j) is outside this range, C(i,j)
% is zero; note that this behavior is an extension to the built-in
% bitget, which results in an error for this case.  This modified rule
% allows the inputs A and B to be sparse.  If B(i,j) is implicitly zero
% (not in the pattern of B), or if A(i,j) is implicitly zero, then C(i,j)
% is not an entry in the pattern of C.
%
% If A and B are matrices, the pattern of C is the set intersection of A
% and B.  If one of A or B is a nonzero scalar, the scalar is expanded
% into a sparse matrix with the same pattern as the other matrix, and the
% result is a sparse matrix.
%
% With a third parameter, C = bitget (A,B,assumedtype) provides a data
% type to convert A to if it has a floating-point type.  If A already has
% an integer type, then it is not modified.  Otherwise, A is converted to
% assumedtype, which can be 'int8', 'int16', 'int32', 'int64', 'uint8',
% 'uint16', 'uint32' or 'uint64'.  The default is 'uint64'.
%
% Example:
%
%   A = GrB (magic (4)'*137, 'uint16')
%   B = GrB (magic (4))
%   C = bitget (A, B)
%   fprintf ('\nA: ') ; fprintf ('%3x ', A) ; fprintf ('\n') ;
%   fprintf ('\nB: ') ; fprintf ('%3x ', B) ; fprintf ('\n') ;
%   fprintf ('\nC: ') ; fprintf ('%3x ', C) ; fprintf ('\n') ;
%   C2 = bitget (uint16 (A), uint16 (B))
%   isequal (C2, C)
%
% See also GrB/bitor, GrB/bitand, GrB/bitxor, GrB/bitcmp, GrB/bitshift,
% GrB/bitset, GrB/bitclr.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 3)
    assumedtype = 'uint64' ;
end

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

atype = gbtype (A) ;
btype = gbtype (B) ;

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

% determine the type of A
if (isequal (atype, 'double') || isequal (atype, 'single'))
    A = gbnew (A, assumedtype) ;
    atype = assumedtype ;
end

% ensure B has the right type
if (~isequal (btype, atype))
    B = gbnew (B, atype) ;
end

% extract the bits from each entry of A
C = GrB (gb_emult (A, ['bitget.' atype], B), ctype) ;

