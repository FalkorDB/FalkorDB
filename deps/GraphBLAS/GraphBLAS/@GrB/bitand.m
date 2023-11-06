function C = bitand (A, B, assumedtype)
%BITAND bitwise AND.
% C = bitand (A,B) is the bitwise AND of A and B.  If A and B are
% matrices, the pattern of C is the set intersection of A and B.  If one
% of A or B is a nonzero scalar, the scalar is expanded into a sparse
% matrix with the same pattern as the other matrix, and the result is a
% sparse matrix.
%
% With a third parameter, C = bitand (A,B,assumedtype) provides a data
% type to convert A and B to if they are floating-point types.  If A or B
% already have integer types, then they are not modified.  Otherwise, A or
% B are converted to assumedtype, which can be 'int8', 'int16', 'int32',
% 'int64', 'uint8', 'uint16', 'uint32' or 'uint64'.  The default is
% 'uint64'.
%
% The input matrices must be real, and may be GraphBLAS/built-in
% matrices, in any combination.  C is returned as a GraphBLAS matrix.
% The type of C is given by GrB.optype (A,B), after any conversion to
% assumedtype, if needed.
%
% Example:
%
%   A = GrB (magic (4), 'uint8')
%   B = GrB (13 * eye (4), 'uint8') ;
%   B (3,4) = 107
%   C = bitand (A, B)
%   fprintf ('\nA: ') ; fprintf ('%3x ', A) ; fprintf ('\n') ;
%   fprintf ('\nB: ') ; fprintf ('%3x ', B) ; fprintf ('\n') ;
%   fprintf ('\nC: ') ; fprintf ('%3x ', C) ; fprintf ('\n') ;
%   C2 = bitand (uint8 (A), uint8 (B))
%   isequal (C2, C)
%
% See also GrB/bitor, GrB/bitxor, GrB/bitcmp, GrB/bitshift, GrB/bitget,
% GrB/bitset, GrB/bitclr.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 3)
    assumedtype = 'uint64' ;
end

C = GrB (gb_bitwise ('bitand', A, B, assumedtype)) ;

