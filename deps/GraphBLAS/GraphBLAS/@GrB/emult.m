function C = emult (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.EMULT sparse element-wise 'multiplication'.
%
%   C = GrB.emult (op, A, B, desc)
%   C = GrB.emult (Cin, accum, op, A, B, desc)
%   C = GrB.emult (Cin, M, op, A, B, desc)
%   C = GrB.emult (Cin, M, accum, op, A, B, desc)
%
% GrB.emult computes the element-wise 'multiplication' T=A.*B.  The result
% T has the pattern of the intersection of A and B. The operator is used
% where A(i,j) and B(i,j) are present.  Otherwise the entry does not
% appear in T.
%
%   if (A(i,j) and B(i,j) is present)
%       T(i,j) = op (A(i,j), B(i,j))
%
% T is then accumulated into C via C<#M,replace> = accum (C,T).
%
% Cin, M, accum, and the optional descriptor desc are the same as all other
% GrB.methods; see GrB.mxm and GrB.descriptorinfo for more details.  For the
% binary operator, see GrB.binopinfo.
%
% See also GrB.eadd.

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
    case 3
        [C, k] = gbemult (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbemult (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbemult (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbemult (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbemult (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
end

if (k == 0)
    C = GrB (C) ;
end

