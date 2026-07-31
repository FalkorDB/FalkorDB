function C = rdivide (A, B)
%RDIVIDE C = A./B, sparse matrix element-wise division.
% C = A./B when B is a matrix results in a full matrix C, with all entries
% present.  If A is a matrix and B is a scalar, then C has the pattern of A,
% except if B is zero and A is double, single, or complex.  In that case, since
% 0/0 is NaN, C is a full matrix.
%
% See also GhB/ldivide, GhB.emult, GhB.eadd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_rdivide (1, A, B) ;

