function C = bitshift (A, B, assumedtype)
%BITSHIFT bitwise left and right shift.
% C = bitshift (A,B) is the bitwise shift of A; if B > 0 then A is shifted left
% by B bits, and if B < 0 then A is shifted right by -B bits.  If either A or B
% are scalars, they are expanded to the pattern of the other matrix.  C has the
% pattern of A (after expansion, if needed).
%
% With a third parameter, C = bitshift (A,B,assumedtype) provides a data type
% to convert A to if it is a floating-point type.  If A already has an integer
% type, then it is not modified.  Otherwise, A is converted to assumedtype,
% which can be 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32'
% or 'uint64'.  The default is 'uint64'.
%
% Example:
%
%   A = uint8 (magic (4))
%   G = GhB (magic (4), 'uint8') ;
%   C1 = bitshift (A, -2) ;
%   C2 = bitshift (G, -2)
%   isequal (C2, C)
%
% See also GhB/bitor, GhB/bitand, GhB/bitxor, GhB/bitcmp, GhB/bitget,
% GhB/bitset, GhB/bitclr.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 2)
    C = gb_bitwise (1, 'bitshift', A, B) ;
else
    C = gb_bitwise (1, 'bitshift', A, B, assumedtype) ;
end

