function testc4(use_builtin)
%TESTC4 test complex extractElement and setElement

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

if (nargin < 1)
    use_builtin = true ;
end
GB_builtin_complex_set (use_builtin) ;

seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        seed = seed + 1 ;
        A = GB_mex_random (m, n, 10*(m+n), 1, seed) ;
        S = logical (spones (A)) ;

        ktuples = 400 ;

        for trials = 1:10

            J0 = irand (0, n-1, ktuples, 1) ;
            I0 = irand (0, m-1, ktuples, 1) ;
            J = double (J0+1) ;
            I = double (I0+1) ;

            use_scalar = (rand (1) > 0.8) ;
            x1 = GB_mex_Matrix_extractElement (A, I0, J0, '', use_scalar)  ;
            s1 = GB_mex_Matrix_isStoredElement (A, I0, J0) ;
            x2 = complex (zeros (ktuples,1)) ;
            s2 = false (ktuples,1) ;
            for k = 1:ktuples
                x2 (k) = A (I (k), J (k)) ;
                s2 (k) = S (I (k), J (k)) ;
            end
            assert (isequal (x1, x2))
            assert (isequal (s1, s2))

            if (n == 1)
                x1 = GB_mex_Vector_extractElement (A, I0, '', use_scalar)  ;
                assert (isequal (x1, x2))
            end
        end
    end
end
fprintf ('All complex extractElement x = A(i,j) tests passed\n') ;


seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        seed = seed + 1 ;
        A = GB_mex_random (m, n, 10*(m+n), 1, seed) ;

        ktuples = 40 ;

        for trials = 1:10

            J0 = irand (0, n-1, ktuples, 1) ;
            I0 = irand (0, m-1, ktuples, 1) ;
            J = double (J0+1) ;
            I = double (I0+1) ;
            X = complex (rand (ktuples,1) + 1i*rand(ktuples,1)) ;

            C1 = GB_mex_setElement (A, I0, J0, X)  ;
            C3 = GB_mex_setElement (A, I0, J0, X, false, true)  ;

            C2 = A ;
            for k = 1:ktuples
                C2 (I (k), J (k)) = X (k) ;
            end
            assert (isequal (C1.matrix, C2))
            assert (isequal (C3.matrix, C2))

        end
    end
end
fprintf ('All complex setElement A(i,j) = x tests passed\n') ;

fprintf ('\ntestc4: all tests passed\n') ;

GB_builtin_complex_set (true) ;
