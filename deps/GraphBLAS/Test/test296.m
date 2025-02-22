function test296
%TEST296 test integer overflow in GB_AxB_saxpy3_cumsum.c
%
% See also test_saxpy3.m

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 1000 ;
nz = 1000 ;
range = int8 ([0 128]) ;

d = nz / n^2 ;
A = sprand (n, n, d) ;
Cin = sparse (n, n) ;
save = GB_mex_hack ;
hack = save ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

for k = 1:2

    if (k == 2)
        hack (5) = 1 ;
        GB_mex_hack (hack) ;
    end

    C1 = GB_mex_mxm (Cin, [ ], [ ], semiring, A, A, [ ]) ;
    C2 = A*A ;
    assert (isequal (C1.matrix, C2)) ;

end

% restore the flag
GB_mex_hack (save) ;

fprintf ('test296: all tests passed\n') ;
