function C = select (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.SELECT select entries from a GraphBLAS sparse matrix.
%
% syntax for a new matrix C (for ops with no b):    computation:
% C = GrB.select (op, A, desc)                      % C = op(A)
% C = GrB.select (Cin, accum, op, A, desc)          % C = Cin ; C += op(A)
% C = GrB.select (Cin, M, op, A, desc)              % C = Cin ; C<M> = op(A)
% C = GrB.select (Cin, M, accum, op, A, desc)       % C = Cin ; C<M> += op(A)
%
% syntax for a new matrix C (for ops with b):
% C = GrB.select (op, A, b, desc)                   % C = op(A,b)
% C = GrB.select (Cin, accum, op, A, b, desc)       % C = Cin ; C += op(A,b)
% C = GrB.select (Cin, M, op, A, b, desc)           % C = Cin ; C<M> = op(A,b)
% C = GrB.select (Cin, M, accum, op, A, b, desc)    % C = Cin ; C<M> += op(A,b)
%
% GrB.select selects a subset of entries from the matrix A, based on their
% value or position (shown as op(A) or op(A,b) above).  For example, L =
% GrB.select ('tril', A, 0) returns the lower triangular part of the GraphBLAS
% or built-in matrix A, just like L = tril (A) for a built-in matrix A.  The
% select operators can also depend on the values of the entries.  The b
% parameter is an input scalar, used in many of the select operators.  For
% example, L = GrB.select ('tril', A, -1) is the same as L = tril (A, -1),
% which returns the strictly lower triangular part of A.  The b scalar is
% required for 'tril', 'triu', 'diag', 'offdiag' and the 2-input operators.  It
% must not appear when using the '*0' operators.
%
% The selectop is a string defining the operator:
%
%   operator        built-in equivalent         equivalent strings
%   --------        -----------------           ------------------
%   'tril'          C = tril (A,b)
%   'triu'          C = triu (A,b)
%   'diag'          C = diag (A,b), see note
%   'offdiag'       C = entries not in diag(A,b)
%   'nonzero'       C = A (A ~= 0)              '~=0'
%   'zero'          C = A (A == 0)              '==0'
%   'positive'      C = A (A >  0)              '>0'
%   'nonnegative'   C = A (A >= 0)              '>=0'
%   'negative'      C = A (A <  0)              '<0'
%   'nonpositive'   C = A (A <= 0)              '<=0'
%   '~='            C = A (A ~= b)
%   '=='            C = A (A == b)
%   '>'             C = A (A >  b)
%   '>='            C = A (A >= b)
%   '<'             C = A (A <  b)
%   '<='            C = A (A <= b)
%
% Many of the operations have equivalent synonyms, as listed above.  Note that
% C = GrB.select ('diag',A,b) does not return a vector, but a diagonal matrix,
% instead.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If either
% accum or M is present, then Cin is a required input.  If desc.in0 is
% 'transpose' then A is transposed before applying the operator.  See 'help
% GrB.descriptorinfo' for more details.
%
% The selectop is a required string defining the select operator to use.  All
% operators operate on all types (the select operators do not do any
% typecasting of its inputs).
%
% See also GrB/tril, GrB/triu, GrB/diag, GrB.selectopinfo, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

narginchk (2, 7) ;

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

if (nargin >= 6 && gb_is_grb (arg6))
    arg6 = struct (arg6) ;
end

% arg7: if present, it must be the descriptor

    switch (nargin)
        case 2
            [C_opaque, kind] = gbmex_select (0, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_select (0, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_select (0, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_select (0, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            [C_opaque, kind] = gbmex_select (0, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_select (0, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
    end
    C = gb_mexfunction_result (0, C_opaque, kind) ;

