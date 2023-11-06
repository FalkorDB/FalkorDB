function test47
%TEST47 prformance test of GrB_vxm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
% d = struct ('inp1', 'tran', 'inp0', 'tran') ;
Prob = ssget (939) ;
% Prob = ssget (2662) ;
A = Prob.A ;

n = size (A,1) ;
A (1,2) = 1 ;

w = sparse (n,1) ;
semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;

t3 = 0 ;

Xnz = [ ]  ;
T = [ ]

d2.axb = 'dot' ;

for xnz = [100:100:1000 2000:1000:72000]

    x = sprand (n, 1, xnz/n) ;

    tic
    c1 = GB_mex_vxm (w, [],[], semiring, x, A, [ ]) ;
    t = toc ;

    tic
    c2 = GB_mex_vxm (w, [],[], semiring, x, A, d2) ;
    t2 = toc ;

    tic
    c0 = x'*A ;
    tm = toc ;

    assert (isequal (c0', c1.matrix)) ;

    Xnz = [Xnz nnz(x)] ;
    T = [T t] ;

    fprintf ('%5d : %10g(%s) %10g built-in %10g speedup %10g %10g\n', ...
        nnz(x), t, method (1), t2, tm, tm/t, tm/t2) ;

end

fprintf ('\ntest47: all tests passed\n') ;


