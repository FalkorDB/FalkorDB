function testc8(use_builtin)
%TESTC8 test complex eWiseAdd and eWiseMult

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('testc8: test complex eWiseAdd and eWiseMult\n') ;

if (nargin < 1)
    use_builtin = true ;
end
GB_builtin_complex_set (use_builtin) ;

rng ('default')
seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        
        for trials = 1:100

            A = GB_mex_random (m, n, 10*(m+n), 1, seed) ; seed = seed + 1 ;
            B = GB_mex_random (m, n, 10*(m+n), 1, seed) ; seed = seed + 1 ;
            S = GB_mex_complex (sparse (m,n)) ;

            C1 = GB_mex_complex (A + B) ;
            C2 = GB_mex_Matrix_eWiseAdd (S, [], [], 'plus', A, B, []) ;
            assert (isequal (C1, C2.matrix)) ;

            if (n == 1)
                C2 = GB_mex_Vector_eWiseAdd (S, [], [], 'plus', A, B, []) ;
                assert (isequal (C1, C2.matrix)) ;
            end

            C1 = GB_mex_complex (A .* B) ;
            C2 = GB_mex_Matrix_eWiseMult (S, [], [], 'times', A, B, []) ;
            % drop explicit zeros from C2.matrix:
            assert (isequal (C1, 1*C2.matrix)) ;

            if (n == 1)
                C2 = GB_mex_Vector_eWiseMult (S, [], [], 'times', A, B, []) ;
                assert (isequal (C1, 1*C2.matrix)) ;
            end
        end
    end
end

fprintf ('testc8: all complex eWise tests passed\n') ;

GB_builtin_complex_set (true) ;
