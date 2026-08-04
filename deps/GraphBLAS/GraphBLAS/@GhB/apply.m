function C = apply (arg1, arg2, arg3, arg4, arg5, arg6)
%GHB.APPLY apply a unary operator to a matrix.
%
% syntax to create a new matrix C:                  computation:
% C = GhB.apply (op, A, desc)                       % C = op(A)
% C = GhB.apply (Cin, accum, op, A, desc)           % C = Cin ; C += op(A)
% C = GhB.apply (Cin, M, op, A, desc)               % C = Cin ; C<M> = op(A)
% C = GhB.apply (Cin, M, accum, op, A, desc)        % C = Cin ; C<M> += op(A)
%
% in-place syntax:
% GhB.apply (C, op, A, desc)                        % C = op(A)
% GhB.apply (C, accum, op, A, desc)                 % C += op(A)
% GhB.apply (C, M, op, A, desc)                     % C<M> = op(A)
% GhB.apply (C, M, accum, op, A, desc)              % C<M> += op(A)
%
% GhB.apply applies a unary operator to the entries in the input matrix A,
% which may be a GraphBLAS or built-in matrix (sparse or full).  See 'help
% GrB.unopinfo' for a list of available unary operators.
%
% The op and A arguments are required.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
% For the in-place syntax, the GhB matrix C is modified in-place.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If either
% accum or M is present, then C or Cin is a required input. If desc.in0 is
% 'transpose' then A is transposed before applying the operator, as C<M> =
% accum (C, f(A')) where f(...) is the unary operator.  See 'help
% GrB.descriptorinfo' for more details.
%
% Example:
%
%   C = GhB.random (4, 4, 0.5)
%   GhB.apply (C, 'sqrt', C) ;      % C = sqrt (C), in-place
%   C
%
% See also GhB/apply2, GhB/spfun, GhB.unopinfo, GhB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargout == 0)
    narginchk (3, 6) ;
else
    narginchk (2, 6) ;
end

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (nargin >= 3 && gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

if (nargin >= 4 && gb_is_grb (arg4))
    arg4 = struct (arg4) ;
end

if (nargin >= 5 && gb_is_grb (arg5))
    arg5 = struct (arg5) ;
end

% arg6: if present, it must be the descriptor

if (nargout == 0)
    switch (nargin)
        case 3
            gbmex_apply (1, arg1, arg2, arg3) ;
        case 4
            gbmex_apply (1, arg1, arg2, arg3, arg4) ;
        case 5
            gbmex_apply (1, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            gbmex_apply (1, arg1, arg2, arg3, arg4, arg5, arg6) ;
    end
else
    switch (nargin)
        case 2
            [C_opaque, kind] = gbmex_apply (1, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_apply (1, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_apply (1, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_apply (1, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            [C_opaque, kind] = gbmex_apply (1, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
    end
    C = gb_mexfunction_result (1, C_opaque, kind) ;
end

