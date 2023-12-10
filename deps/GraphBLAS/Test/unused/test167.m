function test167
%TEST167 test C<M>=A*B with very sparse M, different types of A and B

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

d = 0.02 ;
n = 1000 ;

A.matrix = 100 * sprand (n, n, d) ;
A.matrix (1:257,1) = rand (257, 1) ;

B.matrix = 100 * sprand (n, n, d) ;
B.matrix (1,1) = 1 ;
M = logical (sprand (n, n, 0.002)) ;
Cin.matrix = sparse (n, n) ;

[~, ~, ~, types, ~, ~,] = GB_spec_opsall ;
types = types.all ;

for k = 1:length (types)

    type = types {k} ;
    semiring.class = type ;
    A.class = type ;
    B.class = type ;
    Cin.class = type ;
    fprintf ('%s ', type) ;

    C2 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, [ ]) ;

    if (isequal (type, 'double'))
        A2 = GB_spec_matrix (A) ;
        B2 = GB_spec_matrix (B) ;
        C1 = double (M) .* (A2.matrix * B2.matrix) ;
        err = norm (C1 - C2.matrix, 1) / norm (C1, 1) ;
        assert (err < 1e-12) ;
        % C1 = GB_spec_mxm (Cin, M, [ ], semiring, A, B, [ ]) ;
        % GB_spec_compare (C1, C2) ;
    end
end

fprintf ('\ntest167: all tests passed\n') ;

