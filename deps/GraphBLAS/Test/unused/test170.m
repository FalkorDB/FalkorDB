function test170
%TEST170 test C<B>=A+B (alias M==B)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

fprintf ('test170:\n') ;
GB_mex_burble (1) ;

n = 30 ;

A = GB_spec_random (n, n, 0.5, 1, 'double') ;
A.sparsity = 2 ;    % sparse

B = GB_spec_random (n, n, 0.5, 1, 'double') ;
B.sparsity = 2 ;    % sparse

C1 = spones (B.matrix) .* (A.matrix+B.matrix) ;
C2 = GB_mex_AplusB_M_aliased (A, B, 'plus') ;
GB_spec_compare (C1, C2) ;

GB_mex_burble (0) ;
fprintf ('test170: all tests passed\n') ;

