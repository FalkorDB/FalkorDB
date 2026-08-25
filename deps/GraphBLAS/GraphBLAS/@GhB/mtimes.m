function C = mtimes (A, B)
%MTIMES sparse matrix-matrix multiplication over the standard semiring.
% C=A*B multiples two matrices using the standard '+.*' semiring.  If either A
% or B are scalars, C=A*B is the same as C=A.*B.
%
% See also GhB.mxm, GhB.emult, GhB/times.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gzb_mtimes (1, A, B) ;

