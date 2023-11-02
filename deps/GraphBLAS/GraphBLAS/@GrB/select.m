function C = select (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.SELECT: select entries from a GraphBLAS sparse matrix.
%
%   C = GrB.select (selectop, A)
%   C = GrB.select (selectop, A, b)
%   C = GrB.select (selectop, A, b, desc)
%
%   C = GrB.select (Cin, accum, selectop, A)
%   C = GrB.select (Cin, accum, selectop, A, b)
%   C = GrB.select (Cin, accum, selectop, A, b, desc)
%
%   C = GrB.select (Cin, M, selectop, A)
%   C = GrB.select (Cin, M, selectop, A, b)
%   C = GrB.select (Cin, M, selectop, A, b, desc)
%
%   C = GrB.select (Cin, M, accum, selectop, A)
%   C = GrB.select (Cin, M, accum, selectop, A, b)
%   C = GrB.select (Cin, M, accum, selectop, A, b, desc)
%
% GrB.select selects a subset of entries from the matrix A, based on their
% value or position.  For example, L = GrB.select ('tril', A, 0) returns
% the lower triangular part of the GraphBLAS or built-in matrix A, just
% like L = tril (A) for a built-in matrix A.  The select operators can
% also depend on the values of the entries.  The b parameter is an input
% scalar, used in many of the select operators.  For example, L =
% GrB.select ('tril', A, -1) is the same as L = tril (A, -1), which
% returns the strictly lower triangular part of A.  The b scalar is
% required for 'tril', 'triu', 'diag', 'offdiag' and the 2-input
% operators.  It must not appear when using the '*0' operators.
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
% Note that C = GrB.select ('diag',A,b) does not return a vector,
% but a diagonal matrix, instead.
%
% Many of the operations have equivalent synonyms, as listed above.
%
% Cin is an optional input matrix.  If Cin is not present or is an empty
% matrix (Cin = [ ]) then it is implicitly a matrix with no entries, of
% the right size (which depends on A, and the descriptor).  Its type is
% the output type of the accum operator, if it is present; otherwise, its
% type is the type of the matrix A.
%
% M is the optional mask matrix.  If not present, or if empty, then no
% mask is used.  If present, M must have the same size as C.
%
% If accum is not present, then the operation becomes C<...> =
% select(...).  Otherwise, accum (C, select(...)) is computed.  The accum
% operator acts like a sparse matrix addition (see GrB.eadd).
%
% The selectop is a required string defining the select operator to use.
% All operators operate on all types (the select operators do not do any
% typecasting of its inputs).
%
% A is the input matrix.  It is transposed on input if desc.in0 =
% 'transpose'.
%
% desc is optional. See 'help GrB.descriptorinfo' for more details.
%
% See also GrB/tril, GrB/triu, GrB/diag.

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
    case 2
        [C, k] = gbselect (arg1, arg2) ;
    case 3
        [C, k] = gbselect (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbselect (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbselect (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbselect (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbselect (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
end

if (k == 0)
    C = GrB (C) ;
end

