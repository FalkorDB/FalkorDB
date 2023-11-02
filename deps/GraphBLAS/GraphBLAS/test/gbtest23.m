function gbtest23
%GBTEST23 test min and max

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
for trial = 1:10
    fprintf ('.') ;
    for m = 1:5
        for n = 1:5

            MA = sprand (m, n, 0.5) ;
            S = -(sprand (m, n, 0.5) > 0.5) ;
            MA = MA .* S ;

            MB = sprand (m, n, 0.5) ;
            S = -(sprand (m, n, 0.5) > 0.5) ;
            MB = MB .* S ;

            GA = GrB (MA) ;
            GB = GrB (MB) ;

            c1 = max (MA) ;
            c2 = max (GA) ;
            assert (gbtest_eq (c1, c2)) ;

            c1 = min (MA) ;
            c2 = min (GA) ;
            assert (gbtest_eq (c1, c2)) ;

            C1 = max (MA,MB) ;
            C2 = max (MA,GB) ;
            C3 = max (GA,MB) ;
            C4 = max (GA,GB) ;
            assert (gbtest_eq (C1, C2)) ;
            assert (gbtest_eq (C1, C3)) ;
            assert (gbtest_eq (C1, C4)) ;

            C1 = min (MA,MB) ;
            C2 = min (MA,GB) ;
            C3 = min (GA,MB) ;
            C4 = min (GA,GB) ;
            assert (gbtest_eq (C1, C2)) ;
            assert (gbtest_eq (C1, C3)) ;
            assert (gbtest_eq (C1, C4)) ;

            % c1 = max (MA, [ ], 'all') ;
            c1 = max (max (MA)) ;
            c2 = max (GA, [ ], 'all') ;
            assert (gbtest_eq (c1, c2)) ;

            % c1 = min (MA, [ ], 'all') ;
            c1 = min (min (MA)) ;
            c2 = min (GA, [ ], 'all') ;
            assert (gbtest_eq (c1, c2)) ;

            C1 = max (MA, [ ], 1) ;
            C2 = max (GA, [ ], 1) ;
            assert (gbtest_eq (C1, C2)) ;

            C1 = min (MA, [ ], 1) ;
            C2 = min (GA, [ ], 1) ;
            assert (gbtest_eq (C1, C2)) ;

            C1 = max (MA, [ ], 2) ;
            C2 = max (GA, [ ], 2) ;
            assert (gbtest_eq (C1, C2)) ;

            C1 = min (MA, [ ], 2) ;
            C2 = min (GA, [ ], 2) ;
            assert (gbtest_eq (C1, C2)) ;

        end
    end
end

fprintf ('\ngbtest23: all tests passed\n') ;

