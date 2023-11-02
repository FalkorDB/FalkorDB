function test247
%TEST247 test saxpy3 fine-hash method

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 1000000 ;
A.matrix = sparse (n, n) ;
B.matrix = sprand (n, 1, 0.01) ;
A.matrix (1:100, 1:100) = sprand (100, 100, 0.4) ;
S = sparse (n, 1) ;

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;

[nth chk] = nthreads_get ;

desc.axb = 'hash' ;
nthreads_set (16, 1) ;
GB_mex_burble (1) ;
C1 = GB_mex_mxm (S, [ ], [ ], semiring, A, B, desc) ;
GB_mex_burble (0) ;

C2 = A.matrix * B.matrix ;
err = norm (C1.matrix - C2, 1) ;
fprintf ('err: %g\n', err) ;
assert (err < 1e-12) ;

nthreads_set (nth, chk) ;

fprintf ('\ntest247: all tests passed\n') ;


