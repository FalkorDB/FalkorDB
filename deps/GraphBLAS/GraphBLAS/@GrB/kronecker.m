function C = kronecker (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.KRONECKER sparse Kronecker product.
%
%   C = GrB.kronecker (op, A, B, desc)
%   C = GrB.kronecker (Cin, accum, op, A, B, desc)
%   C = GrB.kronecker (Cin, M, op, A, B, desc)
%   C = GrB.kronecker (Cin, M, accum, op, A, B, desc)
%
% GrB.kronecker computes the Kronecker product, T=kron(A,B), using the
% given binary operator op, in place of the conventional '*' operator for
% the built-in kron.  See also C = kron (A,B), which uses the
% default semiring operators if A and/or B are GrB matrices.
%
% T is then accumulated into C via C<#M,replace> = accum (C,T).
%
% See also kron, GrB/kron.

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
        [C, k] = gbkronecker (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbkronecker (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbkronecker (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbkronecker (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbkronecker (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
end

if (k == 0)
    C = GrB (C) ;
end

