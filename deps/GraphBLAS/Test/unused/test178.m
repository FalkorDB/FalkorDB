function test178
%TEST178 matrix realloc

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test178: --------------------------------- matrix realloc\n') ;

n = 20 ;

rng ('default') ;

desc = struct ('mask', 'complement') ;

for trial = 1:10

    Cin = GB_spec_random (n, n, inf, 1, 'double') ;
    Cin.sparsity = 2 ; % sparse
    M = sparse (n,n) ;
    M (1,1) = 1 ;
    A = sparse (n,n) ;

    C1 = GB_spec_assign (Cin, M, [ ], A, [ ], [ ], desc, false) ;
    C2 = GB_mex_assign  (Cin, M, [ ], A, [ ], [ ], desc) ;
    GB_spec_compare (C1, C2) ;
    sparse (C1.matrix)
    sparse (C2.matrix)

end


fprintf ('\ntest178: all tests passed\n') ;

