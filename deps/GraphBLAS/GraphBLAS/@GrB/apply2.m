function C = apply2 (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.APPLY2 apply a binary operator to a matrix, with scalar binding.
%
%   C = GrB.apply2 (op, A, B)
%   C = GrB.apply2 (op, A, B, desc)
%   C = GrB.apply2 (Cin, accum, op, A, B, desc)
%   C = GrB.apply2 (Cin, M, op, A, B, desc)
%   C = GrB.apply2 (Cin, M, accum, op, A, B, desc)
%
% GrB.apply2 applies a binary operator op(A,B) to a matrix, with one of the
% inputs being the matrix and the other input is bound to a scalar.  See
% 'help GrB.binopinfo'.
%
% The op, A, and B arguments are required.  One of A or B must be a scalar.
% If a scalar is sparse with no entries, it is treated as the value zero.
%
% accum: a binary operator to accumulate the results.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If
% either accum or M is present, then Cin is a required input.  If B is the
% scalar and desc.in0 is 'transpose' then A is transposed before applying
% the operator.  If A is the scalar and desc.in1 is 'transpose.', then the
% input matrix B is transposed before applying the operator.
%
% See also GrB/apply, GrB/spfun.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (arg1))
    arg1 = arg1.opaque ;
end

if (isobject (arg2))
    arg2 = arg2.opaque ;
end

if (isobject (arg3))
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
    case 3
        [C, k] = gbapply2 (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbapply2 (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbapply2 (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbapply2 (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbapply2 (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
end

if (k == 0)
    C = GrB (C) ;
end

