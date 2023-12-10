function s = gbtest_eq (A, B)
%GBTEST_EQ tests if A and B are equal, after dropping zeros.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = isequal (GrB.prune (A), GrB.prune (B)) ;

