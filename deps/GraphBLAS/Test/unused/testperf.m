function testperf
%TESTPERF run all performance tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

t = tic ;
fprintf ('\ntestperf:  run all performance tests\n') ;

test_other ;    % install required packages

debug_off

test26(1) ;     % performance test for GxB_select
test36 ;        % performance test of matrix subref
test30 ;        % performance test GB_mex_subassign, scalar expansion
test30b ;       % performance test GB_mex_assign, scalar expansionb
test35 ;        % performance test for GrB_extractTuples
test39 ;        % performance test for GrB_transpose
test42 ;        % performance tests for GB_mex_Matrix_build
test43 ;        % performance tests for GB_mex_Matrix_subref
test46 ;        % performance test GB_mex_subassign
test48 ;        % performance test of GrB_mxm
test46b ;       % performance test GB_mex_assign
test49 ;        % performance test of GrB_mxm (dot product method, A'*B)
test51 ;        % performance test GB_mex_subassign, multiple ops
test58(0)       % longer GB_mex_Matrix_eWiseAdd performance test
test61 ;        % performance test of GrB_eMult
test68 ;        % performance tests for eWiseMult
f = [936 2662] ;
test73 ;        % performance of C = A*B, with mask
test86 ;        % performance of GrB_Matrix_extract
test52 ;        % performance of A*B with tall matrices, AdotB, AxB

test51b ;       % performance of GrB_assign, multiply operations
test87 ;        % performance test of GrB_mxm
test89 ;        % performance test of complex A*B
test91 ;        % test subref performance on dense vectors
test95 ;        % performance test for GrB_transpose

test111 ;       % performance test for eWiseAdd
test113 ;       % performance tests for GrB_kron
test114 ;       % performance of reduce-to-scalar
test116 ;       % performance tests C(I,J)=A and C=A(I,J)
test117 ;       % performance tests C(:,:)<M> += A
test118 ;       % performance tests C(:,:)<M> = A
test119 ;       % performance tests C(I,J) += scalar
test120 ;       % performance tests C(I,J)<!M> += scalar
test121 ;       % performance tests C(I,J)+=A
test122 ;       % performance tests C(I,J)<!M> += A

% perfoance test of GrB_mxm on all semirings (just auto method)
test06(936, [ ], 1, 0) ;

fprintf ('\ntestperf:  all tests passed.  Total te %g\n', toc (t)) ;

