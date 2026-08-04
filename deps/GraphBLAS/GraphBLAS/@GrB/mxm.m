function C = mxm (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.MXM sparse matrix-matrix multiplication.
%
% syntax for a new matrix C:                        computation:
% C = GrB.mxm (semiring, A, B, desc)                % C = A*B
% C = GrB.mxm (Cin, accum, semiring, A, B, desc)    % C = Cin + A*B
% C = GrB.mxm (Cin, M, semiring, A, B, desc)        % C = Cin ; C<M> = A*B
% C = GrB.mxm (Cin, M, accum, semiring, A, B, desc) % C = Cin ; C<M> += A*B
%
% GrB.mxm computes T = A*B using a given semiring, where C(i,j) =
% sum (A(i,:).*B(:,j).'), except that "sum" can be any monoid, and "*" can be
% any binary operator.
%
% T is then accumulated into C via C<M> = accum (C,T), where the accum step is
% computed using GrB.eadd and M can be modified by the descriptor desc.
%
% The semiring is a required string defining the semiring to use, in the form
% 'add.mult.type', where '.type' is optional.  For example, '+.*.double' is the
% conventional semiring for numerical linear algebra, used in the built-in
% C=A*B when A and B are double.  If A or B are double complex, then C=A*B uses
% the '+.*.double complex' semiring.  GraphBLAS has many more semirings.  See
% 'help GrB.semiringinfo' for more details.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If either
% accum or M is present, then C or Cin is a required input.  If desc.in0 is
% 'transpose' then A is transposed before applying the operator.  If desc.in1
% is 'transpose', then the input matrix B is transposed before applying the
% operator.  See 'help GrB.descriptorinfo' for more details.
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
% See also GrB.descriptorinfo, GrB.eadd, GrB/mtimes, GrB.semiringinfo,
% GrB.moniodinfo, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

narginchk (3, 7) ;

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

if (nargin >= 4 && gb_is_grb (arg4))
    arg4 = struct (arg4) ;
end

if (nargin >= 5 && gb_is_grb (arg5))
    arg5 = struct (arg5) ;
end

if (nargin >= 6 && gb_is_grb (arg6))
    arg6 = struct (arg6) ;
end

% arg7: if present, it must be the descriptor

    switch (nargin)
        case 3
            [C_opaque, kind] = gbmex_mxm (0, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_mxm (0, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_mxm (0, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            [C_opaque, kind] = gbmex_mxm (0, arg1, arg2, arg3, arg4, arg5, ...
                arg6) ;
        case 7
            [C_opaque, kind] = gbmex_mxm (0, arg1, arg2, arg3, arg4, arg5, ...
                arg6, arg7) ;
    end
    C = gb_mexfunction_result (0, C_opaque, kind) ;

