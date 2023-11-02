function test144
%TEST144 test GB_cumsum

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test144 ---------------------- test GB_cumsum\n') ;

rng ('default') ;

n = 1e6 ;

c = int64 (50 * rand (1,n)) ;

for nthreads = 1:8

    % fprintf ('nthreads: %d\n', nthreads) ;

    for malloc_debug_count = 0:2

        [p1, k1] = GB_mex_cumsum (c, nthreads, malloc_debug_count) ;
        p = cumsum ([0 c]) ;
        k = sum (c ~= 0) ;
        assert (isequal (p, p1)) ;
        assert (k == k1) ;

        p1 = GB_mex_cumsum (c, nthreads, malloc_debug_count) ;
        p = cumsum ([0 c]) ;
        assert (isequal (p, p1)) ;

    end
end

fprintf ('test144: all tests passed\n') ;

