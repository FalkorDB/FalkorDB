function C = mxm (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.MXM sparse matrix-matrix multiplication.
%
% GrB.mxm computes C<M> = accum (C, A*B) using a given semiring.
%
% Usage:
%
%   C = GrB.mxm (semiring, A, B)
%   C = GrB.mxm (semiring, A, B, desc)
%
%   C = GrB.mxm (Cin, accum, semiring, A, B)
%   C = GrB.mxm (Cin, accum, semiring, A, B, desc)
%
%   C = GrB.mxm (Cin, M, semiring, A, B)
%   C = GrB.mxm (Cin, M, semiring, A, B, desc)
%
%   C = GrB.mxm (Cin, M, accum, semiring, A, B)
%   C = GrB.mxm (Cin, M, accum, semiring, A, B, desc)
%
% Cin is an optional input matrix.  If Cin is not present or is an empty
% matrix (Cin = [ ]) then it is implicitly a matrix with no entries, of the
% right size (which depends on A, B, and the descriptor).  Its type is the
% output type of the accum operator, if it is present; otherwise, its type
% is the type of the additive monoid of the semiring.
%
% M is the optional mask matrix.  If not present, or if empty, then no mask
% is used.  If present, M must have the same size as C.
%
% If accum is not present, then the operation becomes C<...> = A*B.
% Otherwise, accum (C,A*B) is computed.  The accum operator acts like a
% sparse matrix addition (see GrB.eadd).
%
% The semiring is a required string defining the semiring to use, in the
% form 'add.mult.type', where '.type' is optional.  For example,
% '+.*.double' is the conventional semiring for numerical linear algebra,
% used in the built-in C=A*B when A and B are double.  If A or B are
% double complex, then C=A*B uses the '+.*.double complex' semiring.
% GraphBLAS has many more semirings.  See 'help GrB.semiringinfo' for more
% details.
%
% A and B are the input matrices.  A is transposed on input if desc.in0
% is 'transpose', and/or desc.in1 = 'transpose' transposes B.
%
% desc is optional.  See 'help GrB.descriptorinfo' for more details.
%
% Examples:
%
%   A = sprand (4,5,0.5) ;
%   B = sprand (5,3,0.5) ;
%   C = GrB.mxm ('+.*', A, B) ;
%   norm (C-A*B,1)
%   E = sprand (4,3,0.7) ;
%   M = logical (sprand (4,3,0.5)) ;
%   C2 = GrB.mxm (E, M, '+', '+.*', A, B) ;
%   C3 = E ; AB = A*B ; C3 (M) = C3 (M) + AB (M) ;
%   norm (C2-C3,1)
%
% See also GrB.descriptorinfo, GrB.add, GrB/mtimes.

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
        [C, k] = gbmxm (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbmxm (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbmxm (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbmxm (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbmxm (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
end

if (k == 0)
    C = GrB (C) ;
end

