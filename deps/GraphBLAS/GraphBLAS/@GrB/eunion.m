function C = eunion (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
%GRB.EUNION sparse matrix union.
%
%   C = GrB.eunion (op, A, a, B, b)
%   C = GrB.eunion (op, A, a, B, b, desc)
%   C = GrB.eunion (Cin, accum, op, A, a, B, b, desc)
%   C = GrB.eunion (Cin, M, op, A, a, B, b, desc)
%   C = GrB.eunion (Cin, M, accum, op, A, a, B, b, desc)
%
% GrB.euion computes the element-wise 'addition' T=A+B.  The result T has
% the pattern of the union of A and B. The operator is used for all entries
% in C(i,j), where a and b are scalars:
%
%   if (A(i,j) and B(i,j) is present)
%       T(i,j) = op (A(i,j), B(i,j))
%   elseif (A(i,j) is present but B(i,j) is not)
%       T(i,j) = op (A(i,j), b)
%   elseif (B(i,j) is present but A(i,j) is not)
%       T(i,j) = op (a, B(i,j))
%
% T is then accumulated into C via C<#M,replace> = accum (C,T).
%
% Cin, M, accum, and the descriptor desc are the same as all other
% GrB.methods; see GrB.mxm and GrB.descriptorinfo for more details.  For
% the binary operator, see GrB.binopinfo.
%
% See also GrB.emult.

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

if (isobject (arg4))
    arg4 = arg4.opaque ;
end

if (isobject (arg5))
    arg5 = arg5.opaque ;
end

if (nargin > 5 && isobject (arg6))
    arg6 = arg6.opaque ;
end

if (nargin > 6 && isobject (arg7))
    arg7 = arg7.opaque ;
end

if (nargin > 7 && isobject (arg8))
    arg8 = arg8.opaque ;
end

switch (nargin)
    case 5
        [C, k] = gbeunion (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbeunion (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbeunion (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
    case 8
        [C, k] = gbeunion (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) ;
    case 9
        [C, k] = gbeunion (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) ;
end

if (k == 0)
    C = GrB (C) ;
end

