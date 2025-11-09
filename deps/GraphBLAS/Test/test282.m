function test282
%TEST282 test argmax with index binary op

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n--- testing argmax with user-defined index binary op\n') ;
rng ('default') ;

n = 100 ;
d = 0.1 ;
A = sprand (n, n, d) ;
A = A + pi * speye (n) ;

for jit = 0:1
    [x1, p1] = GB_mex_argmax (A, 1, 1, jit) ;
    [x2, p2] = max (A, [ ], 1) ;
    assert (isequal (x1, x2')) ;
    assert (isequal (p1, p2')) ;
    [x1, p1] = GB_mex_argmax (A, 2, 0, jit) ;
    [x2, p2] = max (A, [ ], 2) ;
    assert (isequal (x1, x2)) ;
    assert (isequal (p1, p2)) ;
end

fprintf ('\ntest282: all tests passed\n') ;

