function testc2(quick,use_builtin)
%TESTC2 test complex A*B, A'*B, A*B', A'*B', A+B

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

if (nargin < 2)
    use_builtin = true ;
end
GB_builtin_complex_set (use_builtin) ;

if (nargin < 1)
    quick = 0 ;
end
if (quick)
    nlist = [1 4] ;
else
    nlist = [1 5 10 100] ;
end

maxerr = 0 ;
for m = nlist
    for n = nlist
        for k = nlist
            A = GB_mex_random (k, m, m*5, 1, 1) ;
            B = GB_mex_random (k, n, n*5, 1, 2) ;
            C = GB_mex_AdotB (A, B) ;
            C2 = A'*B  ;
            err = norm (C-C2,1) ;
            maxerr = max (maxerr, err) ;
            assert (err < 1e-12)
        end
    end
end
fprintf ('All complex A''*B tests passed, maxerr %g\n', maxerr) ;

maxerr = 0 ;
for m = nlist
    for n = nlist
            A = GB_mex_random (m, n, m*5, 1, 1) ;
            B = GB_mex_random (m, n, n*5, 1, 2) ;
            C = GB_mex_AplusB (A, B, 'plus') ;
            C2 = A + B  ;
            err = norm (C-C2,1) ;
            maxerr = max (maxerr, err) ;
            assert (err < 1e-12)
    end
end
fprintf ('All complex A+B tests passed, maxerr %g\n', maxerr) ;

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'complex' ;
dtn.inp0 = 'tran' ;

anum = [0 7081 7083 7084] ;
algos = {'auto', 'gustavson', 'dot', 'hash'} ;

seed = 1 ;

maxerr = 0 ;
for m = nlist
    for n = nlist
        for k = nlist
            for at = 0:1
                for bt = 0:1
                    if (at)
                        A = GB_mex_random (k, m, m*5, 1, seed) ;
                    else
                        A = GB_mex_random (m, k, m*5, 1, seed) ;
                    end
                    seed = seed + 1 ;
                    if (bt)
                        B = GB_mex_random (n, k, m*5, 1, seed) ;
                    else
                        B = GB_mex_random (k, n, m*5, 1, seed) ;
                    end
                    seed = seed + 1 ;

                    if (m == 100 & k > 3 & n > 3)
                        na = size (A,1) ;
                        A (:,1) = 1i * rand (na,1) ;
                        A (:,2:3) = 0 ;
                        A (1,3) = 1 ;
                        nb = size (B,2) ;
                        B (1,:) = 1i * rand (1,nb) ;
                        B (:,2:3) = 0 ;
                        B (2,3) = 4 ;
                    end

                    for aa = 1:length(algos)

                        C = GB_mex_AxB (A, B, at, bt, anum (aa)) ;

                        desc = struct ;
                        if (at)
                            desc.inp0 = 'tran' ;
                        end
                        if (bt)
                            desc.inp1 = 'tran' ;
                        end
                        desc.algo = algos {aa} ;

                        if (at)
                            if (bt)
                                C2 = A'*B'  ;
                            else
                                C2 = A'*B  ;
                            end
                        else
                            if (bt)
                                C2 = A*B'  ;
                            else
                                C2 = A*B  ;
                            end
                        end
                        err = norm (C-C2,1) ;
                        maxerr = max (maxerr, err) ;
                        assert (err < 1e-12)

                    end

                    if (at && ~bt)
                        M = sparse (m, n) ;
                        M(1,1) = 1 ;
                        C2 = GB_mex_AdotB (A, B, M, false) ;
                        C3 = GB_mex_AdotB (A, B, M, true) ;
                        C0 = (A'*B) .* M ;
                        assert (norm (C0-C2,1) < 1e-12) ;
                        assert (norm (C0-C3,1) < 1e-12) ;
                    end
                end
            end
        end
    end
end

fprintf ('testc2: all complex A*B, A''*B, A*B'', A''*B'' tests passed, maxerr %g\n', maxerr) ;

GB_builtin_complex_set (true) ;

