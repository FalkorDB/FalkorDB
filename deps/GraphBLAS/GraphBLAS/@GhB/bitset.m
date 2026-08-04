function C = bitset (A, B, arg3, arg4)
%BITSET set bit.
% C = bitset (A,B) sets a bit in A to 1, where the bit position is determined
% by B.  A is an integer array.  If B(i,j) is an integer in the range 1 (the
% least significant bit) to the number of bits in the data type of A, then
% C(i,j) is equal to the value of A(i,j) after setting the bit to 1.  If B(i,j)
% is outside this range, C(i,j) is set to A(i,j), unmodified; note that this
% behavior is an extension of the built-in bitset, which results in an error
% for this case.  This modified rule allows the inputs A and B to be sparse.
%
% If A and B are matrices, the pattern of C is the set union of A and B.  If
% one of A or B is a nonzero scalar, the scalar is expanded into a sparse
% matrix with the same pattern as the other matrix, and the result is a sparse
% matrix.
%
% If the last input argument is a string, C = bitset (A,B,assumedtype) provides
% a data type to convert A to if it has a floating-point type.  If A already
% has an integer type, then it is not modified.  Otherwise, A is converted to
% assumedtype, which can be 'int8', 'int16', 'int32', 'int64', 'uint8',
% 'uint16', 'uint32' or 'uint64'.  The default is 'uint64'.
%
% C = bitset (A,B,V) sets the bit in A(i,j) at position B(i,j) to 0 if V(i,j)
% is zero, or to 1 if V(i,j) is nonzero.  If V is a scalar, it is implicitly
% expanded to V * spones (B).
%
% All four arguments may be used, as C = bitset (A,B,V,assumedtype).
%
% Example:
%
%   A = GhB (magic (4), 'uint8')
%   B = reshape ([1:8 1:8], 4, 4)
%   C = bitset (A, B)
%   fprintf ('\nA: ') ; fprintf ('%3x ', A) ; fprintf ('\n') ;
%   fprintf ('\nB: ') ; fprintf ('%3x ', B) ; fprintf ('\n') ;
%   fprintf ('\nC: ') ; fprintf ('%3x ', C) ; fprintf ('\n') ;
%   C2 = bitset (uint8 (A), B)
%   isequal (C2, C)
%
% See also GhB/bitor, GhB/bitand, GhB/bitxor, GhB/bitcmp, GhB/bitshift,
% GhB/bitset, GhB/bitclr.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

switch (nargin)
    case 2
        C = gb_bitset (1, A, B) ;
    case 3
        C = gb_bitset (1, A, B, arg3) ;
    case 4
        C = gb_bitset (1, A, B, arg3, arg4) ;
end

