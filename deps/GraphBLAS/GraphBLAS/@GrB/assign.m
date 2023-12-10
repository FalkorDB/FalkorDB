function C = assign (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.ASSIGN: assign a submatrix into a matrix.
%
%   C = GrB.assign (Cin, M, accum, A, I, J, desc)
%
%   C<M>(I,J) = A or accum (C(I,J), A)
%
%   Cin and A are required parameters.  All others are optional.
%   The arguments are parsed according to their type.  Arguments
%   with different types can appear in any order:
%
%       Cin, M, A:  2 or 3 GraphBLAS/built-in sparse/full matrices.
%                   The first three matrix inputs are Cin, M, and A.
%                   If 2 matrix inputs are present, they are Cin and A.
%       accum:      an optional string
%       I,J:        cell arrays:  with no cell inputs: I = { } and J = { }.
%                   with one cell input, I is present and J = { }.
%                   with two cell inputs, I is the first cell input and J
%                   is the second cell input.
%       desc:       an optional struct (must appear as the last argument)
%
% desc: see 'help GrB.descriptorinfo' for details.
%
% I and J are cell arrays.  I contains 0, 1, 2, or 3 items:
%
%       0:  { }     This is ':', like C(:,J), refering to all m
%                   rows, if C is m-by-n.
%
%       1:  { I }   1D list of row indices, like C(I,J).
%
%       2:  { start,fini }  start and fini are scalars (either double,
%                   int64, or uint64).  This defines I = start:fini in
%                   built-in index notation.
%
%       3:  { start,inc,fini } start, inc, and fini are scalars (double,
%                   int64, or uint64).
%
%       The J argument is identical, except that it is a list of column
%       indices of C.  If only one cell array is provided, J = {  } is
%       implied, refering to all n columns of C, like C(I,:).
%       GrB.assign does not support linear indexing of a 2D matrix,
%       as in C(I)=A when C is a 2D matrix.
%
%       If neither I nor J are provided on input, then this implies both
%       I = { } and J = { }, or C(:,:), refering to all rows and columns
%       of C.
%
%       desc.base modifies how I, start, and fini are interpretted.
%       If desc.base is 'zero-based' then they are interpretted as
%       zero-based indices, where 0 is the first row or column.
%       If desc.base is 'one-based' (which is the default), then
%       indices are intrepetted as 1-based.
%
% A: this argument either has size length(I)-by-length(J) (or A' if d.in0
%       is 'transpose'), or it is 1-by-1 for scalar assignment (like
%       C(1:2,1:2)=pi, which assigns the scalar pi to the leading 2-by-2
%       submatrix of C).  For scalar assignment, A must contain an entry;
%       it cannot be empty (for example, A = sparse (0)).
%
% accum: an optional binary operator, defined by a string ('+.double') for
%       example.  This allows for C(I,J) = C(I,J) + A to be computed.  If
%       not present, no accumulator is used and C(I,J)=A is computed.
%
% M: an optional mask matrix, the same size as C.
%
% Cin: a required input matrix, containing the initial content of the
% matrix C.
%
% All input matrices may be either GraphBLAS/built-in matrices, in
% any combination.  C is returned as a GraphBLAS matrix.
%
% Example:
%
%   A = sprand (5, 4, 0.5)
%   AT = A'
%   M = sparse (rand (4, 5)) > 0.5
%   Cin = sprand (4, 5, 0.5)
%
%   d.in0 = 'transpose'
%   d.mask = 'complement'
%   C = GrB.assign (Cin, M, A, d)
%   C2 = Cin
%   C2 (~M) = AT (~M)
%   C2 - sparse (C)
%
%   I = [2 1 5]
%   J = [3 3 1 2]
%   B = sprandn (length (I), length (J), 0.5)
%   Cin = sprand (6, 3, 0.5)
%   C = GrB.assign (Cin, B, {I}, {J})
%   C2 = Cin
%   C2 (I,J) = B
%   C2 - sparse (C)
%
% See also GrB.subassign, GrB/subsasgn.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (arg1))
    arg1 = arg1.opaque ;
end

if (isobject (arg2))
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
    case 2
        [C, k] = gbassign (arg1, arg2) ;
    case 3
        [C, k] = gbassign (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbassign (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbassign (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbassign (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbassign (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
end

if (k == 0)
    C = GrB (C) ;
end

