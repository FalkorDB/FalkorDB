function test184
%TEST184 test special cases for mxm, transpose, and build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
nthreads = nthreads_set ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;
dtn = struct ('inp0', 'tran') ;

A = sprand (0, 10, 1) ;
B = sprand (0, 10, 1) ;
Cin = sparse (10, 10) ;
C1 = A'*B ;
C2 = GB_mex_mxm (Cin, [ ], [ ], semiring, A, B, dtn) 

%----------------------------------------------------------------------

m = 1201 ;
n = 4 ;
k = 26 ;
d = 0.02 ;
A = sprand (m, n, d) ;
B = sprand (m, k, 1.0) ;

C1 = A'*B ;
Cin = sparse (n, k) ;

C2 = GB_mex_mxm (Cin, [ ], [ ], semiring, A, B, dtn) ;

assert (norm (C1 - C2.matrix, 1) < 1e-12)

m = 1048576 ;
n = 20 ;
d = 0.0031119 ;
A = sprand (m, n, d) ;
B = sprand (m, n, d) ;

C1 = A'*B ;
Cin = sparse (n, n) ;

C2 = GB_mex_mxm (Cin, [ ], [ ], semiring, A, B, dtn) ;

assert (norm (C1 - C2.matrix, 1) < 1e-12)

%----------------------------------------------------------------------

n = 1 ;
m = 400 ;
k = 4000 ;
nthreads_set (1) ;
A = sprand (m, k, 0.5) ;
B = sprand (k, n, 0.5) ;

C1 = A*B ;
Cin = sparse (m, n) ;

C2 = GB_mex_mxm (Cin, [ ], [ ], semiring, A, B, [ ]) ;
err = norm (C1 - C2.matrix, 1) ;
assert (err < 1e-10)

%----------------------------------------------------------------------

nthreads_set (4) ;
m = 262144 ;
n = 1048576 ;
d = 4e-6 ;
A = sprand (m, n, d) ;
Cin = sparse (n, m) ;

C1 = A' ;
C2 = GB_mex_transpose (Cin, [ ], [ ], A, [ ]) ;

assert (norm (C1 - C2.matrix, 1) < 1e-12)

[I,J,X] = find (A) ;
[m,n] = size (A) ;
nz = length (I) ;
I = I (randperm (nz)) ;
J = J (randperm (nz)) ;
X = X (randperm (nz)) ;
I0 = uint64 (I) - 1 ;
J0 = uint64 (J) - 1 ;

A1 = sparse (I, J, X, m, n) ;
A2 = GB_mex_Matrix_build (I0, J0, X, m, n, [ ]) ;

assert (norm (A1 - A2.matrix, 1) < 1e-12)

%----------------------------------------------------------------------

v1 = sparse (I, 1, X, m, 1) ;
v2 = GB_mex_Vector_build (I0, X, m, [ ]) ;
assert (norm (v1 - v2.matrix, 1) / norm (v1,1) < 1e-12)

%----------------------------------------------------------------------

% restore # of threads
nthreads_set (nthreads) ;

fprintf ('\ntest184: all tests passed\n') ;

