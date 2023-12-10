function test148
%TEST148 eWiseAdd with aliases

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test148 ---------------eWiseAdd with alias\n') ;

rng ('default') ;

n = 5 ;
C = sprand (n, n, 0.5) ;
A = sprand (n, n, 0.5) ;
M = sprand (n, n, 0.5) ;

C0 = C + A + A ;
C1 = GB_mex_Matrix_eWiseAdd (C, [ ], 'plus', 'plus', A, A, [ ]) ;
assert (norm (C0 - C1.matrix, 1) < 1e-12) ;

C2 = GB_mex_ewise_alias2 (C, 'plus', A, [ ]) ;
assert (norm (C0 - C2.matrix, 1) < 1e-12) ;

C0 = C + A ;
C2 = GB_mex_ewise_alias1 (C, 'plus', A, [ ]) ;
assert (norm (C0 - C2.matrix, 1) < 1e-12) ;

C2 = GB_mex_ewise_alias3 (C, 'plus', A, [ ]) ;
assert (norm (C0 - C2.matrix, 1) < 1e-12) ;

clear C A
C.matrix = sparse (rand (n)) ;
C.sparsity = 8 ;
A.matrix = sparse (rand (n)) ;
A.sparsity = 8 ;

C0 = C.matrix + A.matrix + A.matrix ;
C1 = GB_mex_Matrix_eWiseAdd (C, [ ], 'plus', 'plus', A, A, [ ]) ;
assert (norm (C0 - C1.matrix, 1) < 1e-12) ;

C2 = GB_mex_ewise_alias2 (C, 'plus', A, [ ]) ;
assert (norm (C0 - C2.matrix, 1) < 1e-12) ;

C0 = C.matrix + A.matrix ;
C2 = GB_mex_ewise_alias1 (C, 'plus', A, [ ]) ;
assert (norm (C0 - C2.matrix, 1) < 1e-12) ;

C2 = GB_mex_ewise_alias3 (C, 'plus', A, [ ]) ;
assert (norm (C0 - C2.matrix, 1) < 1e-12) ;

desc = struct ('mask', 'structural') ;

C1 = GB_mex_Matrix_eWiseAdd (C, M, [ ], 'plus', M, M, desc) ;
C2 = GB_mex_ewise_alias4 (C, M, 'plus', desc) ;
assert (norm (C1.matrix - C2.matrix, 1) < 1e-12) ;

clear M
M.matrix = sparse (true (n)) ;
M.sparsity = 8 ;
C1 = GB_mex_Matrix_eWiseAdd (C, M, [ ], 'plus', M, M, desc) ;
C2 = GB_mex_ewise_alias4 (C, M, 'plus', desc) ;
assert (norm (C1.matrix - C2.matrix, 1) < 1e-12) ;

% #define USAGE "C = GB_mex_ewise_alias5 (C, M, op, A, desc)"

C1 = GB_mex_Matrix_eWiseAdd (C, M, [ ], 'plus', A, M, desc) ;
C2 = GB_mex_ewise_alias5 (C, M, 'plus', A, desc) ;
assert (norm (C1.matrix - C2.matrix, 1) < 1e-12) ;

% C<M> = A+M
clear A
A.matrix = sparse (rand (n)) ;
A.sparsity = 8 ;
clear M
M = sprand (n, n, 0.05) ;
C1 = GB_mex_Matrix_eWiseAdd (C, M, [ ], 'plus', A, M, desc) ;
C2 = GB_mex_ewise_alias5 (C, M, 'plus', A, desc) ;
assert (norm (C1.matrix - C2.matrix, 1) < 1e-12) ;

% C<M> = M+A
C1 = GB_mex_Matrix_eWiseAdd (C, M, [ ], 'plus', M, A, desc) ;
C2 = GB_mex_ewise_alias6 (C, M, 'plus', A, desc) ;
assert (norm (C1.matrix - C2.matrix, 1) < 1e-12) ;

fprintf ('test148: all tests passed\n') ;
