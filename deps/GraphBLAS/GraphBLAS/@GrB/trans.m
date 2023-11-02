function C = trans (arg1, arg2, arg3, arg4, arg5)
%GRB.TRANS transpose a sparse matrix.
%
%   C = GrB.trans (A)
%   C = GrB.trans (A, desc)
%   C = GrB.trans (Cin, accum, A, desc)
%   C = GrB.trans (Cin, M, A, desc)
%   C = GrB.trans (Cin, M, accum, A, desc)
%
% The descriptor is optional.  If desc.in0 is 'transpose', then C<M>=A or
% C<M>=accum(C,A) is computed, since the default behavior is to transpose
% the input matrix.
%
% For complex matrices, GrB.trans computes the array transpose, not the
% matrix (complex conjugate) transpose.
%
% See also GrB/transpose, GrB/ctranspose, GrB/conj.

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

switch (nargin)
    case 1
        [C, k] = gbtrans (arg1) ;
    case 2
        [C, k] = gbtrans (arg1, arg2) ;
    case 3
        [C, k] = gbtrans (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbtrans (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbtrans (arg1, arg2, arg3, arg4, arg5) ;
end

if (k == 0)
    C = GrB (C) ;
end

