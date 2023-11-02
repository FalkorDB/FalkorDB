function testc7(use_builtin)
%TESTC7 test complex assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntestc7: all complex assign C(I,J)=A --------------------------\n') ;
rng ('default')

if (nargin < 1)
    use_builtin = true ;
end
GB_builtin_complex_set (use_builtin) ;

dclear.outp = 'replace' ;
dclear.mask = 'complement' ;
tol = 1e-13 ;

seed = 1 ;
for m = [1 5 10 50]
    for n = [1 5 10 50]
        seed = seed + 1 ;
        C = GB_mex_random (m, n, 10*(m+n), 1, seed) ;
        for ni = 1:m
            for nj = 1:n
                I = randperm (m, ni) ;
                J = randperm (n, nj) ;
                seed = seed + 1 ;
                A = GB_mex_random (ni, nj, 2*(ni+nj), 1, seed) ;
                seed = seed + 1 ;
                M = GB_mex_random (ni, nj, 4*(ni+nj), 0, seed) ;
                C1 = C ;
                C1 (I,J) = A ;

                I0 = uint64 (I-1) ;
                J0 = uint64 (J-1) ;

                C2 = GB_mex_subassign (C, [ ], [ ], A, I0, J0, []) ;
                assert (isequal (C1, C2.matrix)) ;

                [C3,c1] = GB_mex_subassign (C, M, [ ], A, I0, J0, [], 'plus') ;
                cin = complex (0,0) ;
                c2 = GB_mex_reduce_to_scalar (cin, '', 'plus', C3) ;
                assert (abs (c1-c2) <= tol * (abs (c1) + 1)) ;

                C1 = C ;
                C1 (I,J) = C1 (I,J) + A ;

                C2 = GB_mex_subassign (C, [ ], 'plus', A, I0, J0, []) ;
                assert (norm (C1 - C2.matrix, 1) <= tol * (norm (C1,1)+1)) ;
                assert (isequal (C1, C2.matrix)) ;

            end
            fprintf ('.') ;
        end

        C = GB_mex_random (m, n, 100*(m*n), 1, seed) ; seed = seed + 1 ;
        M = GB_mex_random (m, n, 4*(ni+nj), 0, seed) ; seed = seed + 1 ;
        A = GB_mex_random (m, n, m+n, 1, seed) ;       seed = seed + 1 ;

        [C3,c1] = GB_mex_subassign (C, M, [ ], A, [ ], [ ], dclear, 'plus') ;
        cin = complex (0,0) ;
        c2 = GB_mex_reduce_to_scalar (cin, '', 'plus', C3) ;
        assert (abs (c1-c2) <= tol * (abs (c1) + 1)) ;

        [C3,c1] = GB_mex_subassign (C, [ ], [ ], A, [ ], [ ], dclear, 'plus') ;
        cin = complex (0,0) ;
        c2 = GB_mex_reduce_to_scalar (cin, '', 'plus', C3) ;
        assert (abs (c1-c2) <= tol * (abs (c1) + 1)) ;

        clear S
        S.matrix = sparse (1i * ones (m,n)) ;
        S.pattern = false (m,n) ;
        cin = complex (1,1) ;
        M = sparse (true (m,n)) ;
        C2 = GB_mex_subassign (S, M, [ ], sparse (cin), ...
            [ ], [ ], struct ('mask', 'structural')) ;
        C1 = sparse (ones (m,n)) ;
        C1 (:,:) = cin ;
        assert (norm (C1-C2.matrix, 1) < 1e-12)

    end
end

fprintf ('\ntestc7: all complex assign C(I,J)=A tests passed\n') ;

GB_builtin_complex_set (true) ;

