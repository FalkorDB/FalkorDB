function test293
%TEST293 merge sort, different integer sizes

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test293 --------------- merge sort, integer cases\n') ;
rng ('default') ;

n = 1e6 ;
imax = 2e6 ;
I = randi (imax, n, 1) ;
J = randi (imax, n, 1) ;
K = (1:n)' ;

for i = [32 64]
    
    if (i == 32)
        I0 = uint32 (I) ;
    else
        I0 = uint64 (I) ;
    end

    for k = [32 64]

        if (k == 32)
            K0 = uint32 (K) ;
        else
            K0 = uint64 (K) ;
        end

        IK1 = sortrows ([I0 K0]) ;
        for nthreads = [1 2 4 8]
            fprintf ('.') ;
            [a b] = GB_mex_msort_2 (I0, K0, nthreads) ;
            assert (isequal (IK1, [a b])) ;
        end

        for j = [32 64]

            if (j == 32)
                J0 = uint32 (J) ;
            else
                J0 = uint64 (J) ;
            end

            IJK = sortrows ([I0 J0 K0]) ;
            for nthreads = [1 2 4 8]
                fprintf ('.') ;
                [a b c] = GB_mex_msort_3 (I0, J0, K0, nthreads) ;
                assert (isequal (IJK, [a b c])) ;
            end
        end
    end
end

fprintf ('\ntest293 --------------- all tests passed\n') ;
