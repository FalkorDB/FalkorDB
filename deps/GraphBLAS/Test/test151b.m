function test151b
%TEST151B test bitshift operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test151b: test bshift operator\n') ;

[~, ~, ~, types, ~, ~,] = GB_spec_opsall ;
types = types.int ;
ops2 = { 'bshift' } ;

int_nbits = [ 8, 16, 32, 64, 8, 16, 32, 64 ] ;

rng ('default') ;
Cin = sparse (4,4) ;
C10 = sparse (10,10) ;
desc.mask = 'complement' ;

for k = 1:8

    type = types {k} ;
    nbits = int_nbits (k) ;
    fprintf ('\n%s', type) ;

    for trial = 1:4
        fprintf ('.') ;

        imax = double (intmax (type) / 4) ;
        A = GB_mex_cast (imax * rand (4), type) ;
        B = GB_mex_cast ((nbits-1) * rand (4), type) + 1 ;
        clear A2 ; A2.matrix = sparse (double (A)) ; A2.class = type ;
        clear B2 ; B2.matrix = sparse (double (B)) ; B2.class = 'int8' ;
        A2.pattern = logical (spones (A)) ;
        B2.pattern = logical (spones (B)) ;
        M = sparse (mod (magic (4), 2)) ;
        clear M2 ; M2.matrix = M ; M2.class = 'logical' ;

        for A_sparsity = [1 2 4 8]
        for B_sparsity = [1 2 4 8]
        for M_sparsity = [1 2 4 8]
        A2.sparsity = A_sparsity ;
        B2.sparsity = B_sparsity ;
        M2.sparsity = M_sparsity ;

        for j = 1:length (ops2)
            opname = ops2 {j} ;
            % C1 = bitop (A, B) ;
            op.opname = opname ; op.optype = type ;

            C1 = GB_spec_Matrix_eWiseMult(Cin, [ ], [ ], op, A2, B2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseMult (Cin, [ ], [ ], op, A2, B2, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseAdd (Cin, [ ], [ ], op, A2, B2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseAdd  (Cin, [ ], [ ], op, A2, B2, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseUnion(Cin, [ ], [ ], op, A2, 3, B2, 2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseUnion (Cin, [ ], [ ], op, A2, 3, B2, 2, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseAdd (Cin, [ ], [ ], op, B2, A2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseAdd  (Cin, [ ], [ ], op, B2, A2, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseUnion(Cin, [ ], [ ], op, B2, 3, A2, 2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseUnion (Cin, [ ], [ ], op, B2, 3, A2, 2, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseMult(Cin, M2, [ ], op, A2, B2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseMult (Cin, M2, [ ], op, A2, B2, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseAdd (Cin, M2, [ ], op, A2, B2, [ ]) ;
            C2 = GB_mex_Matrix_eWiseAdd  (Cin, M2, [ ], op, A2, B2, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseUnion (Cin, M2, [ ], op, A2, 1, B2, 3, [ ]) ;
            C2 = GB_mex_Matrix_eWiseUnion  (Cin, M2, [ ], op, A2, 1, B2, 3, [ ]) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseMult(Cin, M2, [ ], op, A2, B2, desc) ;
            C2 = GB_mex_Matrix_eWiseMult (Cin, M2, [ ], op, A2, B2, desc) ;
            GB_spec_compare (C1, C2) ;

            C1 = GB_spec_Matrix_eWiseAdd (Cin, M2, [ ], op, A2, B2, desc) ;
            C2 = GB_mex_Matrix_eWiseAdd  (Cin, M2, [ ], op, A2, B2, desc) ;
            GB_spec_compare (C1, C2) ;

        end
        end
        end
        end
    end
end

fprintf ('\ntest151b: all tests passed\n') ;

