function test202
%TEST202 test iso add and emult

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% GB_mex_burble (1) ;
n = 10 ;
A.matrix = pi * spones (sprandn (n, n, 0.5)) ;
A.class = 'double' ;
A.iso = true ;

B.matrix = pi * spones (sprandn (n, n, 0.5)) ;
B.class = 'double' ;
B.iso = true ;

op_plus.opname = 'plus' ;
op_plus.optype = 'double' ;

op_first.opname = 'first' ;
op_first.optype = 'double' ;

op_first_single.opname = 'first' ;
op_first_single.optype = 'single' ;

op_second.opname = 'second' ;
op_second.optype = 'double' ;

op_second_single.opname = 'second' ;
op_second_single.optype = 'single' ;

op_max.opname = 'max' ;
op_max.optype = 'double' ;

op_times.opname = 'times' ;
op_times.optype = 'double' ;

op_times_single.opname = 'times' ;
op_times_single.optype = 'single' ;

for A_sparsity = [1 2 4]
    for B_sparsity = [1 2 4]

        A.sparsity = A_sparsity ;
        B.sparsity = B_sparsity ;

        Cin = sparse (n,n) ;
        C1 = GB_mex_Matrix_eWiseAdd (Cin, [ ], [ ], op_plus, A, B, [ ]) ;
        C2 = A.matrix + B.matrix ;
        err = norm (C1.matrix - C2, 1) ;
        assert (err < 1e-12) 
        assert (~C1.iso) ;

        Cin = sparse (n,n) ;
        C1 = GB_mex_Matrix_eWiseMult (Cin, [ ], [ ], op_first_single, A, B, [ ]) ;
        C2 = pi * spones (A.matrix .* B.matrix) ;
        err = norm (C1.matrix - C2, 1) ;
        assert (err < 1e-6) 
        assert (C1.iso) ;

        Cin = sparse (n,n) ;
        C1 = GB_mex_Matrix_eWiseMult (Cin, [ ], [ ], op_first, A, B, [ ]) ;
        C2 = pi * spones (A.matrix .* B.matrix) ;
        err = norm (C1.matrix - C2, 1) ;
        assert (err < 1e-12) 
        assert (C1.iso) ;

        Cin = sparse (n,n) ;
        C1 = GB_mex_Matrix_eWiseMult (Cin, [ ], [ ], op_second_single, A, B, [ ]) ;
        C2 = pi * spones (A.matrix .* B.matrix) ;
        err = norm (C1.matrix - C2, 1) ;
        assert (err < 1e-6) 
        assert (C1.iso) ;

        Cin = sparse (n,n) ;
        C1 = GB_mex_Matrix_eWiseMult (Cin, [ ], [ ], op_second, A, B, [ ]) ;
        C2 = pi * spones (A.matrix .* B.matrix) ;
        err = norm (C1.matrix - C2, 1) ;
        assert (err < 1e-12) 
        assert (C1.iso) ;

        Cin = sparse (n,n) ;
        C1 = GB_mex_Matrix_eWiseMult (Cin, [ ], [ ], op_times_single, A, B, [ ]) ;
        C2 = A.matrix .* B.matrix ;
        err = norm (C1.matrix - C2, 1) ;
        assert (err < 1e-5) 
        assert (C1.iso) ;

        Cin = sparse (n,n) ;
        C1 = GB_mex_Matrix_eWiseMult (Cin, [ ], [ ], op_times, A, B, [ ]) ;
        C2 = A.matrix .* B.matrix ;
        err = norm (C1.matrix - C2, 1) ;
        assert (err < 1e-12) 
        assert (C1.iso) ;

        Cin = sparse (n,n) ;
        C1 = GB_mex_Matrix_eWiseAdd (Cin, [ ], [ ], op_max, A, B, [ ]) ;
        C2 = max (A.matrix, B.matrix) ;
        err = norm (C1.matrix - C2, 1) ;
        assert (err < 1e-12) 
        assert (C1.iso) ;
    end
end

% GB_mex_burble (0) ;
fprintf ('test202: all tests passed\n') ;

