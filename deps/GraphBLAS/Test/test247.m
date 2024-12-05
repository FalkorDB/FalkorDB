function test247
%TEST247 test GrB_mxm (for GB_AxB_saxpy3_fineHash_phase2.c)
%
% This tests the "if (hf == i_unlocked) // f == 2" case in the last block of
% code in GB_AxB_saxpy3_fineHash_phase2.c.  The test is nondeterministic so
% it the test coverage might vary, or even be zero.  See also test246.m.
% It is thus run many times.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

fprintf ('test247: testing of GB_AxB_saxpy3_fineHash_phase2.c\n') ;

for trial = 1:40

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
    C1 = GB_mex_mxm (S, [ ], [ ], semiring, A, B, desc) ;

    C2 = A.matrix * B.matrix ;
    err = norm (C1.matrix - C2, 1) ;
    assert (err < 1e-12) ;

    nthreads_set (nth, chk) ;

    fprintf ('.') ;
end

fprintf ('\ntest247: all tests passed\n') ;


