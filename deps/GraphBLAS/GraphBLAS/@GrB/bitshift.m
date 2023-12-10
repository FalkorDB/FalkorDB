function C = bitshift (A, B, assumedtype)
%BITSHIFT bitwise left and right shift.
% C = bitshift (A,B) is the bitwise shift of A; if B > 0 then A is shifted
% left by B bits, and if B < 0 then A is shifted right by -B bits.  If
% either A or B are scalars, they are expanded to the pattern of the other
% matrix.  C has the pattern of A (after expansion, if needed).
%
% With a third parameter, C = bitshift (A,B,assumedtype) provides a data
% type to convert A to if it is a floating-point type.  If A already has
% an integer type, then it is not modified.  Otherwise, A is converted to
% assumedtype, which can be 'int8', 'int16', 'int32', 'int64', 'uint8',
% 'uint16', 'uint32' or 'uint64'.  The default is 'uint64'.
%
% Example:
%
%   A = uint8 (magic (4))
%   G = GrB (magic (4), 'uint8') ;
%   C1 = bitshift (A, -2) ;
%   C2 = bitshift (G, -2)
%   isequal (C2, C)
%
% See also GrB/bitor, GrB/bitand, GrB/bitxor, GrB/bitcmp, GrB/bitget,
% GrB/bitset, GrB/bitclr.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 3)
    assumedtype = 'uint64' ;
end

C = GrB (gb_bitwise ('bitshift', A, B, assumedtype)) ;

