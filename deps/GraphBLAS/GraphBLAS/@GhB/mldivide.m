function C = mldivide (A, B)
% C = A\B, matrix left division.
% If A is a scalar, then C = A.\B is computed; see 'help ldivide'.  Otherwise,
% C is computed by first converting A and B to built-in sparse matrices, and
% then C=A\B is computed using the built-in backslash.
%
% See also GhB/mrdivide.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_mldivide (1, A, B) ;

