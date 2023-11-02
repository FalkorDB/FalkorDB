function testca(use_builtin)
%TESTCA test complex mxm, mxv, and vxm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    use_builtin = true ;
end
GB_builtin_complex_set (use_builtin) ;

fprintf ('testca: test complex mxm, mxv, and vxm\n') ;
rng ('default') ;
dnn = struct ;
dnt = struct ('inp1', 'tran') ;
dtn = struct ('inp0', 'tran') ;
dtt = struct ('inp0', 'tran', 'inp1', 'tran') ;

algos = {'auto', 'gustavson', 'dot', 'hash', 'saxpy'} ;

for kk = 1:length(algos)
dnn.algo = algos {kk} ;
dnt.algo = algos {kk} ;
dtn.algo = algos {kk} ;
dtt.algo = algos {kk} ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double complex' ;

seed = 1 ;
for m = [1 5 10 100]
    for n = [1 5 10 100]
        for k = [1 5 10 100]
            fprintf ('.') ;
            for trial = 1:31

                A = GB_mex_random (m, k, 10*(m+k), 1, seed) ; seed = seed + 1 ;
                B = GB_mex_random (k, n, 10*(k+n), 1, seed) ; seed = seed + 1 ;
                S = GB_mex_complex (sparse (m,n)) ;
                D = GB_mex_random (m, n, 10*(m+n), 1, seed) ; seed = seed + 1 ;

                if (trial == 31)
                    A (:,1) = 1i * rand (m,1) ;
                    B (1,:) = 1i * rand (1,n) ;
                    if (k > 1)
                        A (:,2) = 0 ;
                        A (1,2) = 1i ;
                        B (2,:) = 0 ;
                        B (2,1) = 1i ;
                    end
                end

                C = A*B ;
                C2 = GB_mex_mxm (S, [], [], semiring, A, B, dnn) ;
                assert (isequal_roundoff (C, C2.matrix)) ;

                if (n == 1)
                    C2 = GB_mex_mxv (S, [], [], semiring, A, B, dnn) ;
                    assert (isequal_roundoff (C, C2.matrix)) ;
                    C2 = GB_mex_vxm (S, [], [], semiring, B, A.', dnn) ;
                    assert (isequal_roundoff (C, C2.matrix)) ;
                end

                if (m > 1 && n > 1)
                    t = min (m,n) ;
                    Eye = speye (m,n) ;
                    C = A*B ;
                    C2 = GB_mex_mxm (S, Eye, [], semiring, A, B, dnn) ;
                    d = full (diag (C)) ;
                    d = GB_mex_complex (sparse (1:t, 1:t, d, m, n)) ;
                    assert (isequal_roundoff (d, C2.matrix)) ;
                end

                C = D + A*B ;
                C2 = GB_mex_mxm (D, [], 'plus', semiring, A, B, dnn) ;
                assert (isequal_roundoff (C, C2.matrix)) ;

                if (n == 1)
                    C2 = GB_mex_mxv (D, [], 'plus', semiring, A, B, dnn) ;
                    assert (isequal_roundoff (C, C2.matrix)) ;
                end

                C2 = GB_mex_mxm (D, [], 'plus', semiring, A, B.', dnt)  ;
                assert (isequal_roundoff (C, C2.matrix)) ;

                C2 = GB_mex_mxm (D, [], 'plus', semiring, A.', B, dtn)  ;
                assert (isequal_roundoff (C, C2.matrix)) ;

                if (n == 1)
                    C2 = GB_mex_mxv (D, [], 'plus', semiring, A.', B, dtn)  ;
                    assert (isequal_roundoff (C, C2.matrix)) ;
                end

                C2 = GB_mex_mxm (D, [], 'plus', semiring, A.', B.', dtt)  ;
                assert (isequal_roundoff (C, C2.matrix)) ;

                M = spones (sprand (m, n, 0.5)) ;
                C2 = GB_mex_mxm (S, M, [ ], semiring, A.', B.', dtt)  ;
                C = (A*B) .* M ;
                assert (isequal_roundoff (C, C2.matrix)) ;

            end
        end
    end
end
end

fprintf ('\ntestca: all complex mxm, mxv, vxm tests passed\n') ;

GB_builtin_complex_set (true) ;

