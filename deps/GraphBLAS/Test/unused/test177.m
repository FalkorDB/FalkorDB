function test177
%TEST177 C<!M>=A*B, C bitmap, M sparse, A bitmap, B sparse

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test177:   C<!M>=A*B, C bitmap, M sparse, A bitmap, B sparse\n') ;

n = 20 ;

rng ('default') ;

desc.mask = 'complement' ;
semiring.class = 'double' ;

for trial = 1:10

    clear M Cin
    M.matrix = logical (spones (sprand (n, n, 0.5))) ;
    M.sparsity = 2 ; % bitmap

    Cin = GB_spec_random (n, n, 0.5, 1, 'double') ;
    Cin.sparsity = 4 ; % bitmap

    A = GB_spec_random (n, n, 0.5, 1, 'double') ;
    A.sparsity = 4 ; % bitmap

    B = GB_spec_random (n, n, 0.5, 1, 'double') ;
    B.sparsity = 2 ; % sparse

    semiring.add = 'plus' ;
    semiring.multiply = 'times' ;

    C1 = GB_spec_mxm (Cin, M, [ ], semiring, A, B, desc) ;
    C2 = GB_mex_mxm  (Cin, M, [ ], semiring, A, B, desc) ;
    GB_spec_compare (C1, C2) ;

    semiring.add = 'max' ;
    semiring.multiply = 'plus' ;

    C1 = GB_spec_mxm (Cin, M, [ ], semiring, A, B, desc) ;
    C2 = GB_mex_mxm  (Cin, M, [ ], semiring, A, B, desc) ;
    GB_spec_compare (C1, C2) ;

end

fprintf ('\ntest177: all tests passed\n') ;

