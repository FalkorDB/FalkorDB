function test96
%TEST96 test dot product

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

n = 1000 ;
A = sprandn (n, n, 0.5) ;
B = sprandn (n, n, 0.5) ;

C1 = A'*B ;

M = spones (sprandn (n, n, 0.5)) ;

C2 = GB_mex_AdotB (A,B) ;

err = norm (C1-C2, 1) / norm (C1, 1)
assert (err < 1e-12)

C4 = C1 .* M ;

C3 = GB_mex_AdotB (A,B,M) ;

err = norm (C3-C4, 1) / norm (C3, 1)
assert (err < 1e-12)

fprintf ('test96: all tests passed\n') ;
