function test152
%TEST152 test C = A+B for dense A, B, and C

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest152: test binops with C=A+B, all dense\n') ;

rng ('default') ;

[binops, ~, ~, types, ~, ~,] = GB_spec_opsall ;
types = types.all ;
binops = binops.all ;

n = 5 ;

Amat5 = 5 * sparse (rand (n)) ;
Bmat5 = 5 * sparse (rand (n)) ;
Amat5 (1,1) = 1 ;
Bmat5 (1,1) = 1 ;
Amat = 20 * Amat5 ;
Bmat = 20 * Bmat5 ;

C.matrix = sparse (ones (n)) ;
C.pattern = sparse (true (n)) ;
A.pattern = sparse (true (n)) ;
A.sparsity = 8 ;
B.pattern = sparse (true (n)) ;
B.sparsity = 8 ;

for k1 = 1:length (binops)
    opname = binops {k1} ;
    fprintf ('\n%-14s ', opname) ;

    for k2 = 1:length (types)
        type = types {k2} ;
        op.opname = opname ;
        op.optype = type ;

        try
            [rename optype ztype xtype ytype] = GB_spec_operator (op) ;
        catch me
            continue ;
        end

        A.class = xtype ;
        B.class = ytype ;
        C.class = ztype ;

        switch (opname)
            case { 'atan2', 'pow' }
                A.matrix = Amat5 ;
                B.matrix = Bmat5 ;
            otherwise
                A.matrix = Amat5 ;
                B.matrix = Bmat5 ;
        end

        if (isequal (opname, 'bitshift') || isequal (opname, 'bshift'))
            B.class = 'int8' ;
        end

        if (test_contains (type, 'single'))
            tol = 1e-5 ;
        elseif (test_contains (type, 'double'))
            tol = 1e-12 ;
        else
            tol = 0 ;
        end

        fprintf ('.') ;

        for C_sparsity = [2 8]
            C.sparsity = C_sparsity ;
            C1 = GB_spec_Matrix_eWiseAdd (C, [ ], [ ], op, A, B, [ ]) ;
            C2 = GB_mex_Matrix_eWiseAdd  (C, [ ], [ ], op, A, B, [ ]) ;
            GB_spec_compare (C1, C2, 0, tol) ;
        end
    end
end

fprintf ('\ntest152: all tests passed\n') ;

