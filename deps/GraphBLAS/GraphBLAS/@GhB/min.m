function C = min (A, B, option)
%MIN Maximum elements of a matrix.
% C = min (A) is the smallest entry in the vector A.  If A is a matrix, C is a
% row vector with C(j) = min (A (:,j)).
%
% C = min (A,B) is an array of the element-wise minimum of two matrices A and
% B, which either have the same size, or one can be a scalar.
%
% C = min (A, [ ], 'all') is a scalar, with the smallest entry in A.
% C = min (A, [ ], 1) is a row vector with C(j) = min (A (:,j))
% C = min (A, [ ], 2) is a column vector with C(i) = min (A (i,:))
%
% The 2nd output of [C,I] = min (...) in the built-in min is not supported; see
% GhB.argmin instead.  The min (..., nanflag) option is not yet supported; only
% the 'omitnan' behavior is supported.
%
% Complex matrices are not supported.
%
% See also GhB/max, GhB.argmin.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

switch (nargin)
    case 1
        C = gb_min (1, A) ;
    case 2
        C = gb_min (1, A, B) ;
    case 3
        C = gb_min (1, A, B, option) ;
end

