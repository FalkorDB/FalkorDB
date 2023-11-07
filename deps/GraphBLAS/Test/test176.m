function test176
%TEST176 test C(I,J)<M,repl> = scalar (method 09, 11), M bitmap

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test176 ------------ assign/subassign: methods 09, 11\n') ;

m = 10 ;
n = 14 ;

rng ('default') ;
desc.outp = 'replace' ;
scalar = sparse (pi) ;
I = [1 2 4 5 6 8] ;
J = [3 2 4 9 7 1] ;
I0 = uint64 (I) - 1 ;
J0 = uint64 (J) - 1 ;

for trial = 1:10

    clear M Cin
    M.matrix = logical (spones (sprand (m, n, 0.5))) ;
    M.sparsity = 4 ; % bitmap

    Cin = GB_spec_random (m, n, 0.5, 1, 'double') ;
    Cin.sparsity = 2 ; % sparse

    C1 = GB_spec_assign (Cin, M, [ ], scalar, I,  J,  desc, true) ;
    C2 = GB_mex_assign  (Cin, M, [ ], scalar, I0, J0, desc) ;
    GB_spec_compare (C1, C2) ;

    C1 = GB_spec_assign (Cin, M, 'plus', scalar, I,  J,  desc, true) ;
    C2 = GB_mex_assign  (Cin, M, 'plus', scalar, I0, J0, desc) ;
    GB_spec_compare (C1, C2) ;

    M.matrix = logical (spones (sprand (length (I), length (J), 0.5))) ;
    M.sparsity = 4 ; % bitmap

    C1 = GB_spec_subassign (Cin, M, [ ], scalar, I,  J,  desc, true) ;
    C2 = GB_mex_subassign  (Cin, M, [ ], scalar, I0, J0, desc) ;
    GB_spec_compare (C1, C2) ;

    C1 = GB_spec_subassign (Cin, M, 'plus', scalar, I,  J,  desc, true) ;
    C2 = GB_mex_subassign  (Cin, M, 'plus', scalar, I0, J0, desc) ;
    GB_spec_compare (C1, C2) ;

end

fprintf ('\ntest176: all tests passed\n') ;
