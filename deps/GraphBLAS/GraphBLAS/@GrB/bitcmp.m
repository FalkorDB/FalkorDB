function C = bitcmp (A, assumedtype)
%BITCMP bitwise complement.
% C = bitcmp (A) is the bitwise complement of A.  C is a full matrix.  To
% complement all the bits in the entries of a sparse matrix, but not the
% implicit entries not in the pattern of C, use C = GrB.apply ('bitcmp', A)
% instead.
%
% With a second parameter, C = bitcmp (A,assumedtype) provides a data type to
% convert A to if it is a floating-point type.  If A already has an integer
% type, then it is not modified.  Otherwise, A is converted to assumedtype,
% which can be 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32'
% or 'uint64'.  The default is 'uint64'.
%
% Example:
%
%   A = GrB (magic (4), 'uint8')
%   C = bitcmp (A)
%   fprintf ('\nA: ') ; fprintf ('%3x ', A) ; fprintf ('\n') ;
%   fprintf ('\nC: ') ; fprintf ('%3x ', C) ; fprintf ('\n') ;
%   C2 = bitcmp (uint8 (A))
%   isequal (C2, C)
%
% See also GrB/bitor, GrB/bitand, GrB/bitxor, GrB/bitshift, GrB/bitget,
% GrB/bitset, GrB/bitclr.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    C = gb_bitcmp (0, A) ;
else
    C = gb_bitcmp (0, A, assumedtype) ;
end

