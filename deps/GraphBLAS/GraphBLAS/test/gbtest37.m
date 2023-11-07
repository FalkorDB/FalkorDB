function gbtest37
%GBTEST37 test istril, istriu, isbanded, isdiag, ishermitian, ...
% issymmetric, bandwith

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
nmax = 5 ;
for trial = 1:10
    fprintf ('.') ;

    for m = 1:nmax
        for n = 1:nmax
            A = sprand (m, n, 0.5) ;

            skew = false ;
            if (m == n)
                if (mod (trial, 10) == 1)
                    % make A symmetric
                    A = A + A' ;
                elseif (mod (trial, 10) == 2)
                    % make A skew symmetric
                    A = A - A' ;
                    skew = true ;
                end
            end

            if (m == n)
                C = A+A' ;
            else
                C = A*A' ;
            end

            if (~skew)
                if (rand < 0.1)
                    A = logical (A) ;
                    C = logical (C) ;
                end
            end

            L = tril (A) ;
            U = triu (A) ;
            D = diag (diag (A)) ;

            GA = GrB (A) ;
            GL = tril (GA) ;
            GU = triu (GA) ;
            GD = diag (diag (GA)) ;
            if (m == n)
                GC = GrB.prune (GA + GA') ;
            else
                GC = GrB.prune (GA * GA') ;
            end

            for fmt = 0:1

                if (fmt == 1)
                    GA = GrB (GA, 'by row') ;
                    GL = GrB (GL, 'by row') ;
                    GU = GrB (GU, 'by row') ;
                    GD = GrB (GD, 'by row') ;
                    GC = GrB (GC, 'by row') ;
                end

                assert (gbtest_eq (A, GA)) ;
                assert (gbtest_eq (L, GL)) ;
                assert (gbtest_eq (U, GU)) ;
                assert (gbtest_eq (D, GD)) ;
                assert (gbtest_eq (C, GC)) ;

                if (~islogical (A))
                    % built-in istril, istriu, and isdiag
                    % are not defined when A is logical.
                    assert (istril (A) == istril (GA)) ;
                    assert (istril (L) == istril (GL)) ;
                    assert (istril (U) == istril (GU)) ;
                    assert (istril (D) == istril (GD)) ;
                    assert (istril (C) == istril (GC)) ;

                    assert (istriu (A) == istriu (GA)) ;
                    assert (istriu (L) == istriu (GL)) ;
                    assert (istriu (U) == istriu (GU)) ;
                    assert (istriu (D) == istriu (GD)) ;
                    assert (istriu (C) == istriu (GC)) ;

                    assert (isdiag (A) == isdiag (GA)) ;
                    assert (isdiag (L) == isdiag (GL)) ;
                    assert (isdiag (U) == isdiag (GU)) ;
                    assert (isdiag (D) == isdiag (GD)) ;
                    assert (isdiag (C) == isdiag (GC)) ;
                end

                assert (ishermitian (A) == ishermitian (GA)) ;
                assert (ishermitian (L) == ishermitian (GL)) ;
                assert (ishermitian (U) == ishermitian (GU)) ;
                assert (ishermitian (D) == ishermitian (GD)) ;
                assert (ishermitian (C) == ishermitian (GC)) ;

                assert (ishermitian (A, 'skew') == ishermitian (GA, 'skew')) ;

                assert (issymmetric (A) == issymmetric (GA)) ;
                assert (issymmetric (L) == issymmetric (GL)) ;
                assert (issymmetric (U) == issymmetric (GU)) ;
                assert (issymmetric (D) == issymmetric (GD)) ;
                assert (issymmetric (C) == issymmetric (GC)) ;

                assert (issymmetric (A, 'skew') == issymmetric (GA, 'skew')) ;

                if (~islogical (A))
                    assert (isequal (bandwidth (A), bandwidth (GA))) ;
                    assert (isequal (bandwidth (L), bandwidth (GL))) ;
                    assert (isequal (bandwidth (U), bandwidth (GU))) ;
                    assert (isequal (bandwidth (D), bandwidth (GD))) ;
                    assert (isequal (bandwidth (C), bandwidth (GC))) ;

                    assert (bandwidth (A, 'lower') == bandwidth (GA, 'lower')) ;
                    assert (bandwidth (L, 'lower') == bandwidth (GL, 'lower')) ;
                    assert (bandwidth (U, 'lower') == bandwidth (GU, 'lower')) ;
                    assert (bandwidth (D, 'lower') == bandwidth (GD, 'lower')) ;
                    assert (bandwidth (C, 'lower') == bandwidth (GC, 'lower')) ;

                    assert (bandwidth (A, 'upper') == bandwidth (GA, 'upper')) ;
                    assert (bandwidth (L, 'upper') == bandwidth (GL, 'upper')) ;
                    assert (bandwidth (U, 'upper') == bandwidth (GU, 'upper')) ;
                    assert (bandwidth (D, 'upper') == bandwidth (GD, 'upper')) ;
                    assert (bandwidth (C, 'upper') == bandwidth (GC, 'upper')) ;

                    for lo = 0:nmax
                        for hi = 0:nmax
                            assert (isbanded (A,  lo, hi) == ...
                                    isbanded (GA, lo, hi)) ;
                            assert (isbanded (L,  lo, hi) == ...
                                    isbanded (GL, lo, hi)) ;
                            assert (isbanded (U,  lo, hi) == ...
                                    isbanded (GU, lo, hi)) ;
                            assert (isbanded (D,  lo, hi) == ...
                                    isbanded (GD, lo, hi)) ;
                        end
                    end
                end
            end
        end
    end
end

fprintf ('\ngbtest37: all tests passed\n') ;

