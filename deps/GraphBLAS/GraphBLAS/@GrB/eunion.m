function C = eunion (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
%GRB.EUNION sparse matrix union.
%
% syntax for a new matrix C:                      computation:
% C = GrB.eunion (op, A,a,B,b, desc)              % C = op(A,a,B,b)
% C = GrB.eunion (Cin, accum, op, A,a,B,b, desc)  % C = Cin + op(A,a,B,b)
% C = GrB.eunion (Cin, M, op, A,a,B,b, desc)      % C = Cin ; C<M> = op(A,a,B,b)
% C = GrB.eunion (Cin, M, accum, op,A,a,B,b,desc) % C = Cin ; C<M>+= op(A,a,B,b)
%
% in-place syntax:
% GrB.eunion (C, op, A,a,B,b, desc)               % C = op(A,a,B,b)
% GrB.eunion (C, accum, op, A,a,B,b, desc)        % C += op(A,a,B,b)
% GrB.eunion (C, M, op, A,a,B,b, desc)            % C<M> = op(A,a,B,b)
% GrB.eunion (C, M, accum, op, A,a,B,b, desc)     % C<M> += op(A,a,B,b)
%
% GrB.eunion computes the element-wise 'addition' T=A+B, using any binary op
% (shown as op(A,a,B,b) in the computations listed above).  The result T has
% the pattern of the union of A and B. The operator is used for all entries in
% T(i,j), where a and b are scalars:
%
%   if (A(i,j) and B(i,j) is present)
%       T(i,j) = op (A(i,j), B(i,j))
%   elseif (A(i,j) is present but B(i,j) is not)
%       T(i,j) = op (A(i,j), b)
%   elseif (B(i,j) is present but A(i,j) is not)
%       T(i,j) = op (a, B(i,j))
%
% T is then accumulated into C via C<M> = accum (C,T), where the accum step
% is computed using GrB.eadd and M can be modified by the descriptor desc.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If either
% accum or M is present, then C or Cin is a required input.  If desc.in0 is
% 'transpose' then A is transposed before applying the operator.  If desc.in1
% is 'transpose', then the input matrix B is transposed before applying the
% operator.
%
% See also GrB.eadd, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

narginchk (5, 9) ;

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

if (gb_is_grb (arg4))
    arg4 = struct (arg4) ;
end

if (gb_is_grb (arg5))
    arg5 = struct (arg5) ;
end

if (nargin >= 6 && gb_is_grb (arg6))
    arg6 = struct (arg6) ;
end

if (nargin >= 7 && gb_is_grb (arg7))
    arg7 = struct (arg7) ;
end

if (nargin >= 8 && gb_is_grb (arg8))
    arg8 = struct (arg8) ;
end

% arg9: if present, it must be the descriptor

    switch (nargin)
        case 5
            [C_opaque, kind] = gbmex_eunion (0, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            [C_opaque, kind] = gbmex_eunion (0, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_eunion (0, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
        case 8
            [C_opaque, kind] = gbmex_eunion (0, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7, arg8) ;
        case 9
            [C_opaque, kind] = gbmex_eunion (0, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7, arg8, arg9) ;
    end
    C = gb_mexfunction_result (0, C_opaque, kind) ;

