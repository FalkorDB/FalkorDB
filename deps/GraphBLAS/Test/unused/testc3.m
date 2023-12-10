function testc3(use_builtin)
%TESTC3 test complex GrB_extract

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

if (nargin < 1)
    use_builtin = true ;
end
GB_builtin_complex_set (use_builtin) ;

d = struct ('outp', 'replace') ;
seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        seed = seed + 1 ;
        A = GB_mex_random (m, n, 10*(m+n), 1, seed) ;
        for trials = 1:10
            j = randperm (n, 1) ;
            j0 = uint64 (j-1) ;
            w = GB_mex_complex (sprandn (m,1,0.1) + 1i*sprandn(m,1,0.1)) ;

            x1 = GB_mex_Col_extract (w, [],[], A, [], j0, d) ;
            x2 = A (:,j) ;
            assert (isequal (x1.matrix, x2))

            x1 = GB_mex_Col_extract (w, [],'plus', A, [], j0, []) ;
            x2 = w + A (:,j) ;
            assert (isequal (x1.matrix, x2))

            if (m > 2)
                I = 1:(m/2) ;
                I0 = uint64 (I-1) ;
                wi = GB_mex_complex (w (I)) ;

                x1 = GB_mex_Col_extract (wi, [],[], A, I0, j0, d) ;
                x2 = A (I,j) ;
                assert (isequal (x1.matrix, x2))

                x1 = GB_mex_Col_extract (wi, [],'plus', A, I0, j0, d) ;
                x2 = GB_mex_complex (wi + A (I,j)) ;
                assert (isequal (x1.matrix, x2))

            end
        end
    end
end
fprintf ('All complex col extract w = A(:,j) tests passed\n') ;


seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        seed = seed + 1 ;
        A = GB_mex_random (m, n, 10*(m+n), 1, seed) ;
        C = GB_mex_complex (sprandn (m,n,0.1) + 1i*sprandn(m,n,0.1)) ;
        S = GB_mex_complex (sparse (m,n)) ;
        for trials = 1:10

            J = randperm (n, 1+floor(n/2)) ;
            I = randperm (m, 1+floor(m/2)) ;
            J0 = uint64 (J-1) ;
            I0 = uint64 (I-1) ;

            x1 = GB_mex_Matrix_extract (S, [],[], A, [], [], []) ;
            x2 = A ;
            assert (isequal (x1.matrix, x2))

            if (n == 1)
                x1 = GB_mex_Vector_extract (S, [],[], A, [], []) ;
                assert (isequal (x1.matrix, x2))
            end

            Sij = GB_mex_complex (S (I,J)) ;

            x1 = GB_mex_Matrix_extract (Sij, [],[], A, I0, J0, []) ;
            x2 = GB_mex_complex (A (I,J)) ;
            assert (isequal (x1.matrix, x2))

            if (n == 1)
                x1 = GB_mex_Vector_extract (Sij, [],[], A, I0, []) ;
                assert (isequal (x1.matrix, x2))
            end

            Cij = GB_mex_complex (C (I,J)) ;

            x1 = GB_mex_Matrix_extract (Cij, [],'plus', A, I0, J0, []) ;
            x2 = GB_mex_complex (Cij + A (I,J)) ;
            assert (isequal (x1.matrix, x2))

            if (n == 1)
                x1 = GB_mex_Vector_extract (Cij, [],'plus', A, I0, []) ;
                assert (isequal (x1.matrix, x2))
            end

        end
    end
end
fprintf ('All complex extract C = A(I,J) tests passed\n') ;

fprintf ('\ntestc3: all tests passed\n') ;

GB_builtin_complex_set (true) ;
