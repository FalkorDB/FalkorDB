function C = apply (arg1, arg2, arg3, arg4, arg5, arg6)
%GRB.APPLY apply a unary operator to a matrix.
%
%   C = GrB.apply (op, A)
%   C = GrB.apply (op, A, desc)
%   C = GrB.apply (Cin, accum, op, A, desc)
%   C = GrB.apply (Cin, M, op, A, desc)
%   C = GrB.apply (Cin, M, accum, op, A, desc)
%
% GrB.apply applies a unary operator to the entries in the input matrix
% A, which may be a GraphBLAS or built-in matrix (sparse or full).
% See 'help GrB.unopinfo' for a list of available unary operators.
%
% The op and A arguments are required.
%
% accum: a binary operator to accumulate the results.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If
% either accum or M is present, then Cin is a required input. If desc.in0
% is 'transpose' then A is transposed before applying the operator, as
% C<M> = accum (C, f(A')) where f(...) is the unary operator.
%
% See also GrB/apply2, GrB/spfun.

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

switch (nargin)
    case 2
        [C, k] = gbapply (arg1, arg2) ;
    case 3
        [C, k] = gbapply (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbapply (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbapply (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbapply (arg1, arg2, arg3, arg4, arg5, arg6) ;
end

if (k == 0)
    C = GrB (C) ;
end

