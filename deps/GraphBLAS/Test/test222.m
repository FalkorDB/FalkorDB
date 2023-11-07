function test222
%TEST222 test user selectop for iso matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

GB_mex_burble (1) ;
n = 10 ;

A.matrix = pi * spones (sprand (n, n, 0.5)) ;

lo = -3 ;
hi = 4 ;

for iso = [false true]
    A.iso = iso ;
    for sparsity = [1 2 4 8]

        if (sparsity == 8)
            A.matrix = pi * sparse (ones (n, n)) ;
        end
        A.sparsity = sparsity ;

        C1 = GB_mex_band (A, lo, hi, 0) ;
        C2 = triu (tril (A.matrix, hi), lo) ;
        C3 = GB_mex_band2 (A, lo, hi, 0) ;
        assert (isequal (C1, C2)) ;
        assert (isequal (C1, C3)) ;

        C1 = GB_mex_band (A, lo, hi, 1) ;
        C2 = triu (tril (A.matrix', hi), lo) ;
        C3 = GB_mex_band2 (A, lo, hi, 1) ;
        assert (isequal (C1, C2)) ;
        assert (isequal (C1, C3)) ;
    end
end
GB_mex_burble (0) ;
fprintf ('\ntest222: all tests passed\n') ;

