function C = extract (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.EXTRACT extract sparse submatrix.
%
%   C = GrB.extract (Cin, M, accum, A, I, J, desc)
%
%   C<M> = A(I,J) or accum (C, A(I,J))
%
% A is a required parameter.  All others are optional, but if M or accum
% appears, then Cin is also required.  If desc.in0 is 'transpose', then
% the description below assumes A = A' is computed first before the
% extraction (A is not changed on output, however).
%
% desc: see 'help GrB.descriptorinfo' for details.
%
% I and J are cell arrays.  I contains 0, 1, 2, or 3 items:
%
%       0:   { }    This is the built-in ':', like A(:,J), refering to
%                   all m rows, if A is m-by-n.
%
%       1:   { I }  1D list of row indices, like A(I,J).
%
%       2:  { start,fini }  start and fini are scalars (either double,
%                   int64, or uint64).  This defines I = start:fini in
%                   index notation.
%
%       3:  { start,inc,fini } start, inc, and fini are scalars (double,
%                   int64, or uint64).  This defines I = start:inc:fini in
%                   notation.
%
%       The J argument is identical, except that it is a list of column
%       indices of A.  If only one cell array is provided, J = {  } is
%       implied, refering to all n columns of A, like A(I,:).
%       GrB.extract does not support linear indexing of a 2D matrix,
%       as in C=A(I) when A is a 2D matrix.
%
%       If neither I nor J are provided on input, then this implies both
%       I = { } and J = { }, or A(:,:) refering to all rows and columns
%       of A.
%
%       If desc.base is 'zero-based', then I and J are interpretted as
%       zero-based, where the rows and columns of A range from 0 to m-1
%       and n-1, respectively.  If desc.base is 'one-based' (which is the
%       default), then indices are intrepetted as 1-based.
%
% Cin: an optional input matrix, containing the initial content of the
%       matrix C.  C on output is the content of C after the assignment is
%       made.  If present, Cin argument has size length(I)-by-length(J).
%       If accum is present then Cin is a required input.
%
% accum: an optional binary operator, defined by a string ('+.double') for
%       example.  This allows for C = Cin + A(I,J) to be computed.  If
%       not present, no accumulator is used and C=A(I,J) is computed.
%       If accum is present then Cin is a required input.
%
% M: an optional mask matrix, the same size as C.
%
% Example:
%
%   A = sprand (5, 4, 0.5)
%   I = [2 1 5]
%   J = [3 3 1 2]
%   C = GrB.extract (A, {I}, {J})
%   C2 = A (I,J)
%   C2 - C
%
% See also GrB/subsref.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (arg1))
    arg1 = arg1.opaque ;
end

if (nargin > 1 && isobject (arg2))
    arg2 = arg2.opaque ;
end

if (nargin > 2 && isobject (arg3))
    arg3 = arg3.opaque ;
end

if (nargin > 3 && isobject (arg4))
    arg4 = arg4.opaque ;
end

if (nargin > 4 && isobject (arg5))
    arg5 = arg5.opaque ;
end

if (nargin > 5 && isobject (arg6))
    arg6 = arg6.opaque ;
end

switch (nargin)
    case 1
        [C, k] = gbextract (arg1) ;
    case 2
        [C, k] = gbextract (arg1, arg2) ;
    case 3
        [C, k] = gbextract (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbextract (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbextract (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbextract (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbextract (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
end

if (k == 0)
    C = GrB (C) ;
end

