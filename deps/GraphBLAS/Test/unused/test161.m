function test161
%TEST161 C=A*B*E

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 100 ;
d = 0.05 ;
semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

for trial = 1:10
    
    A = sprand (n, n, d) ;
    B = sprand (n, n, d) ;
    E = sprand (n, n, d) ;
    semiring.class = 'double' ;

    C1 = A*B*E ;
    C2 = GB_mex_triple_mxm (semiring, A, B, E) ;
    GB_spec_compare (C1, C2) ;
end

