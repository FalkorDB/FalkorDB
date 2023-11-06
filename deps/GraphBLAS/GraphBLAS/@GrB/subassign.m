function C = subassign (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.SUBASSIGN: assign a submatrix into a matrix.
%
%   C = GrB.subassign (Cin, M, accum, A, I, J, desc)
%
%   Cin and A are required parameters.  All others are optional.
%   The arguments are parsed according to their type.  Arguments
%   with different types can appear in any order:
%
%       Cin, M, A:  2 or 3 GraphBLAS or built-in sparse/full matrices.
%                   The first three matrix inputs are Cin, M, and A.
%                   If 2 matrix inputs are present, they are Cin and A.
%       accum:      an optional string
%       I,J:        cell arrays:  with no cell inputs: I = { } and J = { }.
%                   with one cell input, I is present and J = { }.
%                   with two cell inputs, I is the first cell input and J
%                   is the second cell input.
%       desc:       an optional struct (must appear as the last argument)
%
% GrB.subassign is identical to GrB.assign, with two key differences:
%
%   (1) The mask is different.
%       With GrB.subassign, the mask M is length(I)-by-length(J),
%       and M(i,j) controls how A(i,j) is assigned into C(I(i),J(j)).
%       With GrB.assign, the mask M has the same size as C,
%       and M(i,j) controls how C(i,j) is assigned.
%   (2) The d.out = 'replace' option differs.  GrB.assign can clear
%       entries outside the C(I,J) submatrix; GrB.subassign cannot.
%
% If there is no mask, or if I and J are ':', then the two methods are
% identical.  The examples shown in 'help GrB.assign' also work with
% GrB.subassign.  Otherwise, GrB.subassign is faster.  The two methods are
% described below, where '+' is the optional accum operator.
%
%   step  | GrB.assign      GrB.subassign
%   ----  | ----------      -------------
%   1     | S = C(I,J)      S = C(I,J)
%   2     | S = S + A       S<M> = S + A
%   3     | Z = C           C(I,J) = S
%   4     | Z(I,J) = S
%   5     | C<M> = Z
%
% Refer to GrB.assign for a description of the other input/outputs.
%
% See also GrB.assign, GrB/subsasgn.

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
        [C, k] = gbsubassign (arg1, arg2) ;
    case 3
        [C, k] = gbsubassign (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbsubassign (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbsubassign (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbsubassign (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbsubassign (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
end

if (k == 0)
    C = GrB (C) ;
end

