function test179
%TEST179 bitmap select

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test179: --------------------------------- bitmap select\n') ;

ops = { 'nonzero',  'eq_zero', 'ne_thunk', 'eq_thunk' } ;

n = 20 ;
rng ('default') ;

Cin = GB_spec_random (n, n, 0.5, 1, 'double complex') ;
A = GB_spec_random (n, n, 0.5, 1, 'double complex') ;
scalar = 3+1i ;
A.matrix  (2:3,2:4) = scalar ;
A.pattern (2:3,2:4) = true ;

for builtin = 1 % 0:1
    GB_builtin_complex_set (builtin) ;
    for sparsity_control = [1 2 4]
        Cin.sparsity = sparsity_control ;
        A.sparsity = sparsity_control ;
        for k = 1:length (ops)
            op = ops {k} ;
            C1 = GB_spec_select (Cin, [], [], op, A, scalar, []) ;
            C2 = GB_mex_select  (Cin, [], [], op, A, scalar, []) ;
            GB_spec_compare (C1, C2) ;
        end
    end
end

% try a user-defined selectop
Cin = GB_spec_random (n, n, 0.5, 1, 'double') ;
A = GB_spec_random (n, n, 0.5, 1, 'double') ;
A.matrix  (2:3,2:4) = nan ;
A.pattern (2:3,2:4) = true ;
for sparsity_control = [1 2 4]
    Cin.sparsity = sparsity_control ;
    A.sparsity = sparsity_control ;
    C1 = GB_spec_select (Cin, [], [], 'isnan', A, 0, []) ;
    C2 = GB_mex_select  (Cin, [], [], 'isnan', A) ;
    GB_spec_compare (C1, C2) ;
end

GB_builtin_complex_set (true) ;
fprintf ('\ntest179: all tests passed\n') ;

