function C = times (A, B)
%TIMES C = A.*B, sparse matrix element-wise multiplication.
% C = A.*B computes the element-wise multiplication of A and B.  If both A and
% B are matrices, the pattern of C is the intersection of A and B.  If one is a
% scalar, the pattern of C is the same as the pattern of the one matrix.
%
% See also GrB/mtimes, GrB.emult, GrB.mxm.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_emult (0, A, '*', B) ;

