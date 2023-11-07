function gbtest30
%GBTEST30 test colon notation

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
n = 1e8 %#ok<*NOPRT>
A = sparse (n, 1) ;

k = n/2 - 1 ;
A (1) = 1 ; 
A (3) = 42 ;
A (k) = 87 ; 
A (n) = 92 ;

A
G = GrB (A)

% computes z = G (1:2:k), very quickly, without forming
% the explicit vector 1:2:k
tic
z = G ({1,2,k})
toc

% slow, likely because 1:2:k is constructed
tic
x = A (1:2:k)
toc

% computes y = G (1:2:k), but using subsref overloading.
% This is slow, because 1:2:k is constructed for subsref.
tic
y = G (1:2:k)
toc

assert (gbtest_eq (y, z)) ;
assert (gbtest_eq (x, y)) ;

% GraphBLAS can construct huge-by-huge matrices
tic
[c, huge] = computer %#ok<*ASGLU>
huge = 2^48 ;
H = GrB (huge, huge)
I = sort (randperm (huge, 4)) ;
M = magic (4)
H (I,I) = M
M2 = H (I,I)
t = toc ;
fprintf ('\ntime to construct H: %g sec\n', t) ;
assert (gbtest_eq (M, M2))

middle = ceil (median (I))

% this is very fast:
fprintf ('GraphBLAS colon notation:\nmiddle = %g\n\n', middle) ;
fprintf ('H2 = H ({1, middle}, {1, middle}) works, and is very fast:\n') ;
tic
H2 = H ({1, middle}, {1, middle}) %#ok<*NASGU>
toc

% This is not possible, because 1:middle is too big:
try
    fprintf ('H2 = H (1:middle, 1:middle) will fail:\n') ;
    H2 = H (1:middle, 1:middle)
catch expected_error
    fprintf ('colon notation 1:%d fails (too big!)\n\n', middle) ;
    expected_error
end

fprintf ('gbtest30: all tests passed\n') ;

