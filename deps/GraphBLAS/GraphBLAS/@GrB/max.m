function C = max (A, B, option)
%MAX Maximum elements of a matrix.
% C = max (A) is the largest entry in the vector A.  If A is a matrix, C is a
% row vector with C(j) = max (A (:,j)).
%
% C = max (A,B) is an array of the element-wise maximum of two matrices A and
% B, which either have the same size, or one can be a scalar.
%
% C = max (A, [ ], 'all') is a scalar, with the largest entry in A.
% C = max (A, [ ], 1) is a row vector with C(j) = max (A (:,j))
% C = max (A, [ ], 2) is a column vector with C(i) = max (A (i,:))
%
% The 2nd output of [C,I] = max (...) in the built-in max is not supported; see
% GrB.argmax instead.  The max (..., nanflag) not yet supported; only the
% 'omitnan' behavior is supported.
%
% Complex matrices are not supported.
%
% See also GrB/min, GrB.argmax.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

switch (nargin)
    case 1
        C = gb_max (0, A) ;
    case 2
        C = gb_max (0, A, B) ;
    case 3
        C = gb_max (0, A, B, option) ;
end

