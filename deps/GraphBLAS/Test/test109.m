function test109
%TEST109 terminal monoid with user-defined type

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest109: terminal monoid with user-defined type\n') ;

for k = [false true]
    GB_builtin_complex_set (k) ;

    rng ('default') ;

    A = sparse (rand (4) + 1i * rand (4)) ;

    [i j x] = find (A) ;
    s = prod (prod (x)) ;
    t = GB_mex_reduce_complex (A) ;
    err = norm (s-t) ;
    assert (err < 1e-12)

    A (3,1) = 0 ;

    [i j x] = find (A) ;
    s = prod (prod (x)) ;
    t = GB_mex_reduce_complex (A) ;
    err = norm (s-t) ;
    assert (err < 1e-12)

    t = GB_mex_reduce_complex (A, 4) ;
    assert (t == 0)
end

GB_builtin_complex_set (true) ;
fprintf ('\ntest109: all tests passed\n') ;
