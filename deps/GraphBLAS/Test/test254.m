function test254
%TEST254 test masks with all types

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test254 ------------ C<M>=A+B with different masks\n') ;

fprintf ('jit control: %d, factory: %d\n', ...
    GB_mex_jit_control, GB_mex_factory_control) ;
rng ('default') ;

op = 'plus' ;
[~, ~, ~, types, ~, ~, ~,] = GB_spec_opsall ;
types = types.all ;

n = 20 ;
C0 = sparse (n, n) ;
A = GB_spec_random (n, n, 0.5, 100, 'double') ;
B = GB_spec_random (n, n, 0.5, 100, 'double') ;

for k = 1:length(types)
    type = types {k} ;
    M = GB_spec_random (n, n, 0.1, 100, type) ;
    M.matrix = full (M.matrix) ;
    M.matrix = GB_mex_cast (M.matrix, type) ;

    % C<M> A+B
    X2 = GB_mex_Matrix_eWiseAdd  (C0, M, [ ], op, A, B, [ ]) ;
    X1 = GB_spec_Matrix_eWiseAdd (C0, M, [ ], op, A, B, [ ]) ;
    GB_spec_compare (X1, X2) ;
end

fprintf ('test254 all tests passed\n') ;

