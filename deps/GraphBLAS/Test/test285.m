function test285
%TEST285 test GrB_assign (bitmap case, C<!M>+=A, whole matrix)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% tests GB_bitmap_assign_7_whole

rng ('default') ;

desc.mask = 'complement' ;

accum.opname = 'plus' ;
accum.optype = 'double' ;

% I = ":"
I = [ ] ;
I0 = uint64 (I-1) ;

% J = ":"
J = [ ] ;
J0 = uint64 (I-1) ;

A = GB_spec_random (10, 10, 0.2) ;
A.sparsity = 1 ;    % sparse

C = GB_spec_random (10, 10, 0.2) ;
C.sparsity = 4 ;    % bitmap

M = GB_spec_random (10, 10, 0.2) ;
M.sparsity = 1 ;    % sparse

% C<!M>(I,J) = accum (C (I,J),A)
C0 = GB_spec_assign (C, M, accum, A, I, J, desc, false) ;
GrB.burble (1) ;
C1 = GB_mex_assign  (C, M, accum, A, I0, J0, desc) ;
GrB.burble (0) ;
GB_spec_compare (C0, C1) ;

% C<!M>(I,J) = accum (C (I,J),A), with int16 mask matrix M
M.matrix = ceil (M.matrix) ;
M.class = 'int16' ;
C0 = GB_spec_assign (C, M, accum, A, I, J, desc, false) ;
GrB.burble (1) ;
C1 = GB_mex_assign  (C, M, accum, A, I0, J0, desc) ;
GrB.burble (0) ;
GB_spec_compare (C0, C1) ;

% C<!M>(I,J) = accum (C (I,J),A), with int32 mask matrix M
M.matrix = ceil (M.matrix) ;
M.class = 'int32' ;
C0 = GB_spec_assign (C, M, accum, A, I, J, desc, false) ;
GrB.burble (1) ;
C1 = GB_mex_assign  (C, M, accum, A, I0, J0, desc) ;
GrB.burble (0) ;
GB_spec_compare (C0, C1) ;

% C<!M>(I,J) = accum (C (I,J),A), with double complex mask matrix M
M.class = 'double complex' ;
C0 = GB_spec_assign (C, M, accum, A, I, J, desc, false) ;
GrB.burble (1) ;
C1 = GB_mex_assign  (C, M, accum, A, I0, J0, desc) ;
GrB.burble (0) ;
GB_spec_compare (C0, C1) ;

fprintf ('\ntest285: all tests passed\n') ;


