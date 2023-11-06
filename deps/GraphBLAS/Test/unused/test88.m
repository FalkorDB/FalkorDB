function test88
%TEST88 test hypersparse matrices with hash-based method

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
d.axb = 'hash' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;
semiring.add = 'plus' ;

for n = [10 100 200 300 1000]
    
    for trials = 1:100

        A = GB_spec_random (n, n, 0.001, 1, 'double', true, true, 0.001) ;
        B = GB_spec_random (n, n, 0.001, 1, 'double', true, true, 0.001) ;
        S = sparse (n,n) ;

        B.matrix  (1:10,:) = sparse (rand (10,n)) ;
        B.pattern (1:10,:) = sparse (true (10,n)) ;
        B.matrix  (:,10) = sparse (rand (n,1)) ;
        B.pattern (:,10) = sparse (true (n,1)) ;

        C1 = A.matrix * B.matrix ;
        C2 = GB_mex_mxm (S, [ ], [ ], semiring, A, B, d) ;

        assert (isequal_roundoff (C1, C2.matrix)) ;
    end
end

fprintf ('test88: all tests passed\n') ;

