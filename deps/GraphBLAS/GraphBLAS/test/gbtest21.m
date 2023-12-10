function gbtest21
%GBTEST21 test isfinite, isinf, isnan

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
for trial = 1:40
    fprintf ('.') ;
    for m = 0:5
        for n = 0:5
            A = 100 * sprand (m, n, 0.5) ;
            if (rand < 0.1)
                A = int32 (full (A)) ;
            else
                A (1,1) = nan ; %#ok<*SPRIX>
                A (2,2) = inf ;
            end
            G = GrB (A) ;

            assert (gbtest_eq (isfinite (A), isfinite (G))) ;
            assert (gbtest_eq (isinf    (A), isinf    (G))) ;
            assert (gbtest_eq (isnan    (A), isnan    (G))) ;

            assert (isrow    (A) == isrow    (G)) ;
            assert (iscolumn (A) == iscolumn (G)) ;
        end
    end
end

fprintf ('\ngbtest21: all tests passed\n') ;

