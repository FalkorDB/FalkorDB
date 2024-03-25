function test169
%TEST169 C<M>=A+B with C sparse, M hyper, A and B sparse

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

fprintf ('test169:\n') ;

n = 10 ;
C = GB_spec_random (n, n, 0.5, 1, 'double') ;
C.sparsity = 2 ;    % sparse

M = GB_spec_random (n, n, 0.02, 1, 'double') ;
M.sparsity = 1 ;    % hypersparse

A = GB_spec_random (n, n, 0.5, 1, 'double') ;
A.sparsity = 2 ;    % sparse

B = GB_spec_random (n, n, 0.5, 1, 'double') ;
B.sparsity = 2 ;    % sparse

desc = struct ('mask', 'complement') ;

C1 = GB_spec_Matrix_eWiseAdd (C, M, [ ], 'plus', A, B, desc) ;
C2 = GB_mex_Matrix_eWiseAdd  (C, M, [ ], 'plus', A, B, desc) ;
GB_spec_compare (C1, C2) ;

fprintf ('test169: all tests passed\n') ;

