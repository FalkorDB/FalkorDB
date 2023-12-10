function test27
%TEST27 test GxB_select with user-defined select op (LoHi_band)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test 27: GxB_select with user-defined op (LoHi_band)\n') ;

for m = 1:10
    for n = 1:10
        fprintf ('.') ;
        A = sprand (m, n, 0.1) ;
        for lo = -12:12
            for hi = -12:12

                C1 = GB_mex_band (A, lo, hi, 0) ;
                C2 = triu (tril (A,hi), lo) ;
                assert (isequal (C1, C2)) ;

                C1 = GB_mex_band (A, lo, hi, 1) ;
                C2 = triu (tril (A',hi), lo) ;
                assert (isequal (C1, C2)) ;

            end
        end
    end
end

fprintf ('\ntest27: all tests passed\n') ;

