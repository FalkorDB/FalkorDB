function test298
%TEST298 test GrB_assign (method 08n when A is full)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest298: GrB_assign with method 08n when A is full\n') ;

rng ('default') ;
m = 100 ;
n = 200 ;

% create a CSR matrix
C0 = GB_spec_random (m, n, 0.5, 100, 'double', false, false) ;

am = 5 ;
an = 4 ;
A = rand (am, an) ;

J = [3 4 5 6]' ;
J0 = (uint64 (J) - 1) ;
I = [1 2 4 5 3]' ;
I0 = (uint64 (I) - 1) ;

M = GB_spec_random (am, an, 0.5, 1, 'bool') ;

for C_sparsity = [1 2 4 8]
    C0.sparsity = C_sparsity ;
    for M_sparsity = [1 2 4 8]
        M.sparsity = M_sparsity ;
        C1 = GB_mex_subassign  (C0, M, 'plus', A, I0, J0, [ ]) ;
        C2 = GB_spec_subassign (C0, M, 'plus', A, I, J, [ ], 0) ;
        GB_spec_compare (C1, C2) ;
    end
end

fprintf ('\ntest298: all tests passed\n') ;

