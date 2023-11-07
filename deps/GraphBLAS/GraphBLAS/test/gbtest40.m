function gbtest40
%GBTEST40 test sum, prod, max, min, any, all, norm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

x = GrB.random (10, 1, inf, 'range', complex ([0 1])) ;
s1 = norm (x, 2) ;
s2 = norm (double (x), 2) ;
assert (abs (s1-s2) < 1e-12) ;

x = GrB.random (10, 1, inf, 'range', int16 ([1 16])) ;
s1 = norm (x, 2) ;
s2 = norm (double (x), 2) ;
assert (abs (s1-s2) < 1e-6) ;

s1 = norm (x, -inf) ;
s2 = norm (double (x), -inf) ;
assert (abs (s1-s2) < 1e-6) ;

for trial = 1:3
    for m = 1:3
        for n = 1:3
            fprintf ('.') ;
            for d = [0.1 0.5 1]
                for kind = 0:1

                    if (d == 1)
                        A = sparse (rand (m,n)) ;
                    else
                        A = sprand (m, n, d) ;
                    end
                    if (kind == 1)
                        A = logical (A) ;
                    end
                    G = GrB (A) ;

                    s1 = sum (A) ;
                    s2 = sum (G) ;
                    assert (norm (s1-double(s2), 1) < 1e-12) ;
                    s1 = sum (A,1) ;
                    s2 = sum (G,1) ;
                    assert (norm (s1-double(s2), 1) < 1e-12) ;
                    s1 = sum (A,2) ;
                    s2 = sum (G,2) ;
                    assert (norm (s1-double(s2), 1) < 1e-12) ;
                    s1 = sum (sum (A)) ;
                    s2 = sum (G, 'all') ;
                    assert (norm (s1-double(s2), 1) < 1e-12) ;

                    s1 = prod (A) ;
                    s2 = prod (G) ;
                    assert (norm (s1-double(s2), 1) < 1e-12) ;
                    s1 = prod (A,1) ;
                    s2 = prod (G,1) ;
                    assert (norm (s1-double(s2), 1) < 1e-12) ;
                    s1 = prod (A,2) ;
                    s2 = prod (G,2) ;
                    assert (norm (s1-double(s2), 1) < 1e-12) ;
                    s1 = prod (prod (A)) ;
                    s2 = prod (G, 'all') ;
                    assert (norm (s1-double(s2), 1) < 1e-12) ;

                    if (kind == 0)

                        s1 = norm (A,1) ;
                        s2 = norm (G,1) ;
                        assert (abs (s1 - s2) < 1e-12) ;

                        if (isvector (A))
                            s1 = norm (A,2) ;
                            s2 = norm (G,2) ;
                            assert (abs (s1 - s2) < 1e-12) ;
                            s1 = norm (A) ;
                            s2 = norm (G) ;
                            assert (abs (s1 - s2) < 1e-12) ;
                        end

                        s1 = norm (A,inf) ;
                        s2 = norm (G,inf) ;
                        assert (abs (s1 - s2) < 1e-12) ;

                        s1 = norm (A,'fro') ;
                        s2 = norm (G,'fro') ;
                        assert (abs (s1 - s2) < 1e-12) ;

                        if (isvector (A))
                            s1 = norm (A,2) ;
                            s2 = norm (G,2) ;
                            assert (abs (s1 - s2) < 1e-12) ;
                            s1 = norm (A,-inf) ;
                            s2 = norm (G,-inf) ;
                            assert (abs (s1 - s2) < 1e-12) ;
                        end

                        s1 = max (A) ;
                        s2 = max (G) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = max (A, [ ], 1) ;
                        s2 = max (G, [ ], 1) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = max (A, [ ], 2) ;
                        s2 = max (G, [ ], 2) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = max (max (A)) ;
                        s2 = max (G,  [ ], 'all') ;
                        assert (gbtest_eq (s1, s2)) ;

                        s1 = max (A, 0.5) ;
                        s2 = max (G, 0.5) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = max (A, -0.5) ;
                        s2 = max (G, -0.5) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = max (0.3, A) ;
                        s2 = max (0.3, G) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = max (-0.3, A) ;
                        s2 = max (-0.3, G) ;
                        assert (gbtest_eq (s1, s2)) ;

                        s1 = min (A) ;
                        s2 = min (G) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = min (A, [ ], 1) ;
                        s2 = min (G, [ ], 1) ;
                        assert (gbtest_eq (s1, s2)) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = min (min (A)) ;
                        s2 = min (G,  [ ], 'all') ;
                        assert (gbtest_eq (s1, s2)) ;

                        s1 = min (A, 0.5) ;
                        s2 = min (G, 0.5) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = min (A, -0.5) ;
                        s2 = min (G, -0.5) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = min (0.3, A) ;
                        s2 = min (0.3, G) ;
                        assert (gbtest_eq (s1, s2)) ;
                        s1 = min (-0.3, A) ;
                        s2 = min (-0.3, G) ;
                        assert (gbtest_eq (s1, s2)) ;

                    end

                    s1 = any (A) ;
                    s2 = any (G) ;
                    assert (all (s1 == s2)) ;
                    s1 = any (A,1) ;
                    s2 = any (G,1) ;
                    assert (all (s1 == s2)) ;
                    s1 = any (A,2) ;
                    s2 = any (G,2) ;
                    assert (all (s1 == s2)) ;
                    s1 = any (any (A)) ;
                    s2 = any (G, 'all') ;
                    assert (all (s1 == s2)) ;

                    s1 = all (A) ;
                    s2 = all (G) ;
                    assert (all (s1 == s2)) ;
                    s1 = all (A,1) ;
                    s2 = all (G,1) ;
                    assert (all (s1 == s2)) ;
                    s1 = all (A,2) ;
                    s2 = all (G,2) ;
                    assert (all (s1 == s2)) ;
                    s1 = all (all (A)) ;
                    s2 = all (G, 'all') ;
                    assert (all (s1 == s2)) ;

                end
            end
        end
    end
end

fprintf ('\ngbtest40: all tests passed\n') ;

