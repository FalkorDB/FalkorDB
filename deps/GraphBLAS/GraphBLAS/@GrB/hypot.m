function C = hypot (A, B)
%HYPOT robust computation of the square root of sum of squares.
% C = hypot (A,B) computes sqrt (abs (A).^2 + abs (B).^2) accurately.  If A and
% B are matrices, the pattern of C is the set union of A and B.  If one of A or
% B is a nonzero scalar, the scalar is expanded into a full matrix the size of
% the other matrix, and the result is a full matrix.
%
% See also GrB/abs, GrB/norm, GrB/sqrt, GrB/plus, GrB.eadd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_hypot (0, A, B) ;

