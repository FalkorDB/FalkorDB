function C = eadd (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.EADD sparse matrix addition.
%
%   C = GrB.eadd (op, A, B)
%   C = GrB.eadd (op, A, B, desc)
%   C = GrB.eadd (Cin, accum, op, A, B, desc)
%   C = GrB.eadd (Cin, M, op, A, B, desc)
%   C = GrB.eadd (Cin, M, accum, op, A, B, desc)
%
% GrB.eadd computes the element-wise 'addition' T=A+B.  The result T has
% the pattern of the union of A and B. The operator is used where A(i,j)
% and B(i,j) are present.  Otherwise the entries in A and B are copied
% directly into T:
%
%   if (A(i,j) and B(i,j) is present)
%       T(i,j) = op (A(i,j), B(i,j))
%   elseif (A(i,j) is present but B(i,j) is not)
%       T(i,j) = A(i,j)
%   elseif (B(i,j) is present but A(i,j) is not)
%       T(i,j) = B(i,j)
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
        [C, k] = gbeadd (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbeadd (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbeadd (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbeadd (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbeadd (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
end

if (k == 0)
    C = GrB (C) ;
end

