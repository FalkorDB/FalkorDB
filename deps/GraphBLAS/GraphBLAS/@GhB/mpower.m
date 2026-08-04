function C = mpower (A, B)
%A^B matrix power.
% C = A^B computes the matrix power of A raised to the B. A must be a square
% matrix.  B must an integer >= 0.
%
% See also GhB/power.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_mpower (1, A, B) ;

