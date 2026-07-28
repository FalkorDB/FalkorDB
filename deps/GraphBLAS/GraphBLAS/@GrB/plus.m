function C = plus (A, B)
%PLUS sparse matrix addition, C = A+B.
% C = A+B adds the two matrices A and B.  If A and B are matrices, the pattern
% of C is the set union of A and B.  If one of A or B is a nonzero scalar, the
% scalar is expanded into a full matrix the size of the other matrix, and the
% result is a full matrix.
%
% See also GrB.eadd, GrB/minus, GrB/uminus.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_eadd (0, A, '+', B) ;

