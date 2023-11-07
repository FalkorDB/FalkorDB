function gbtest20
%GBTEST20 test bandwidth, isdiag, ceil, floor, round, fix

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
for trial = 1:10
    fprintf ('.') ;
    for m = 0:10
        for n = 0:10
            A = 100 * sprandn (m, n, 0.5) ;
            G = GrB (A) ;
            [lo1, hi1] = bandwidth (A) ;
            [lo2, hi2] = bandwidth (G) ;
            assert (isequal (lo1, lo2)) ;
            assert (isequal (hi1, hi2)) ;
            d1 = isdiag (A) ;
            d2 = isdiag (G) ;
            assert (isequal (d1, d2)) ;

            assert (gbtest_eq (sign  (A), sign  (G))) ;
            assert (gbtest_eq (ceil  (A), ceil  (G))) ;
            assert (gbtest_eq (floor (A), floor (G))) ;
            assert (gbtest_eq (round (A), round (G))) ;
            assert (gbtest_eq (fix   (A), fix   (G))) ;
        end
    end
end

n = 2^60 ;
G = GrB (n, n) ;
G (n,1) = 1
[lo, hi] = bandwidth (G)
assert (lo == int64 (2^60) - 1)

fprintf ('\ngbtest20: all tests passed\n') ;

