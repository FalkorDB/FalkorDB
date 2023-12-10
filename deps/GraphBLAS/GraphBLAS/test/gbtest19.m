function gbtest19
%GBTEST19 test mpower

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
A = rand (4) ;
G = GrB (A) ;
maxerr = 0 ;

for k = 0:10
    C1 = A^k ;
    C2 = G^k ;
    err = norm (C1 - C2, 1) ;
    assert (err < 1e-10) ;
    maxerr = max (maxerr, err) ;
    C1 = A.^k ;
    C2 = G.^k ;
    err = norm (C1 - C2, 1) ;
    assert (err < 1e-10) ;
    maxerr = max (maxerr, err) ;
end

a = pi ;
b = 3.7 ;
c1 = a^b ;
x = GrB (a) ;
c2 = x^b ;
err = abs (c1-c2) ;
assert (err < 1e-10) ;
maxerr = max (maxerr, err) ;

for type = { 'single', 'double' }
    I1 = complex (eye (4, type{1})) ;
    I2 = GrB (I1) ;
    C1 = I1^0 ;
    C2 = I2^0 ;
    assert (norm (C1 - C2, 1) == 0) ;
    C1 = A^0 ;
    C2 = G^0 ;
    assert (norm (C1 - C2, 1) == 0) ;
    C1 = A^1 ;
    C2 = G^1 ;
    assert (norm (C1 - C2, 1) == 0) ;
    C1 = I1^1 ;
    C2 = I2^1 ;
    assert (norm (C1 - C2, 1) == 0) ;
end

fprintf ('max error: %g\n', maxerr) ;
fprintf ('gbtest19: all tests passed\n') ;

