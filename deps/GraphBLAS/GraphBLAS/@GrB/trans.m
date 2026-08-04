function C = trans (arg1, arg2, arg3, arg4, arg5)
%GRB.TRANS transpose a sparse matrix.
%
% syntax for a new matrix C:                    computation:
% C = GrB.trans (A, desc)                       % C = A'
% C = GrB.trans (Cin, accum, A, desc)           % C = Cin + A'
% C = GrB.trans (Cin, M, A, desc)               % C = Cin ; C<M> = A'
% C = GrB.trans (Cin, M, accum, A, desc)        % C = Cin ; C<M> += A'
%
% GrB.trans computes T=A'.  T is then accumulated into C via
% C<M> = accum (C,T), where the accum step is computed using GrB.eadd and M can
% be modified by the descriptor desc.
%
% For complex matrices, GrB.trans computes the array transpose, not the matrix
% (complex conjugate) transpose.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If either
% accum or M is present, then C or Cin is a required input.  If desc.in0 is
% 'transpose' then A is transposed before applying the operator.  See 'help
% GrB.descriptorinfo' for more details.
%
% See also GrB/transpose, GrB/ctranspose, GrB/conj, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

narginchk (1, 5) ;

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (nargin >= 2 && gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (nargin >= 3 && gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

if (nargin >= 4 && gb_is_grb (arg4))
    arg4 = struct (arg4) ;
end

% arg5: if present, it must be the descriptor

    switch (nargin)
        case 1
            [C_opaque, kind] = gbmex_trans (0, arg1) ;
        case 2
            [C_opaque, kind] = gbmex_trans (0, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_trans (0, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_trans (0, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_trans (0, arg1, arg2, arg3, arg4, arg5) ;
    end
    C = gb_mexfunction_result (0, C_opaque, kind) ;

