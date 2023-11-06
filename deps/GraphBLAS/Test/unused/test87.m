function test87
%TEST87 performance test of GrB_mxm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

rng ('default') ;

%-------------------------------------------------------------------------------

fprintf ('\n--------------------------------------------------\n') ;

k = 30e6
fprintf ('building random sparse matrix, %d by %d\n', k,2) ;
A = sprandn (k, 2, 0.01) ;
B = sprandn (k, 2, 0.01) ;

fprintf ('builtin:\n') ;
tic
C = A'*B ;
toc
tm = toc ;

fprintf ('GrB AdotB:\n') ;
tic
C2 = GB_mex_AdotB (A,B) ;
toc

fprintf ('GrB (A'')*B:\n') ;
tic
C3 = GB_mex_AxB (A',B) ;
toc

fprintf ('GrB A''*B native:\n') ;
C4 = GB_mex_AxB (A,B, true) ;
tg = toc ;

assert (norm (C-C2,1) / norm (C,1) < 1e-12)
assert (norm (C-C3,1) / norm (C,1) < 1e-12)
assert (norm (C-C4,1) / norm (C,1) < 1e-12)

fprintf ('builtin: %10.4f  GB:auto: %10.4f speedup %10.4f\n', ...
    tm, tg, tm/tg) ;

%-------------------------------------------------------------------------------
fprintf ('\n--------------------------------------------------\n') ;

k = 30e6
m = 100
fprintf ('building random sparse matrix, %d by %d\n', k,m) ;
A = sprandn (k, 2, 0.01) ;
B = sprandn (k, m, 0.01) ;

fprintf ('builtin:\n') ;
tic
C = A'*B ;
toc
tm = toc ;

fprintf ('GrB AdotB:\n') ;
tic
C2 = GB_mex_AdotB (A,B) ;
toc

fprintf ('GrB (A'')*B:\n') ;
tic
C3 = GB_mex_AxB (A',B) ;
tg1 = toc ;
fprintf ('just A*B %g (both A and B non-hypersparse)\n', tg1) ;

% this is slower than GB_mex_AxB (A',B) even though it uses the
% same method, because the builtin A' above is non-hypersparse,
% but the internal AT=A' is hypersparse.

fprintf ('GrB A''*B native (AT becomes hypersparse):\n') ;
tic
C4 = GB_mex_AxB (A,B, true) ;
tg = toc ;

fprintf ('builtin: %10.4f  GB:auto: %10.4f(%s) speedup %10.4f\n', ...
    tm, tg, tm/tg) ;

assert (norm (C-C2,1) / norm (C,1) < 1e-12)
assert (norm (C-C3,1) / norm (C,1) < 1e-12)
assert (norm (C-C4,1) / norm (C,1) < 1e-12)

%-------------------------------------------------------------------------------
fprintf ('\n--------------------------------------------------\n') ;

fprintf ('matrix A above, transposed:\n') ;

tic
AT1 = A' ;
toc
tm = toc ;

[mm nn] = size (AT1) ;
S = sparse (mm,nn) ;

tic
AT2 = GB_mex_transpose (S, [ ], [ ], A)
tg = toc ;

assert (isequal (AT1, AT2.matrix)) ;

fprintf ('size of AT: %d %d\n', mm, nn) ;
fprintf ('builtin transpose %g GB %g speedup %g\n', tm, tg, tm/tg) ;

fprintf ('GrB (AT)*B:\n') ;
tic
C3 = GB_mex_AxB (AT1,B) ;
tg1 = toc ;
fprintf ('just A*B %g\n', tg1) ;

%-------------------------------------------------------------------------------
fprintf ('\n--------------------------------------------------\n') ;

fprintf ('\nA''*x where A is big and x is a dense vector\n') ;
Prob = ssget (2662) ;
A = Prob.A ;
n = size (A, 1) ;
x = sparse (rand (n,1)) ;
z = full (x) ;

fprintf ('builtin: x full:\n') ;
tic
y0 = A'*z ;
toc

fprintf ('builtin: x sparse:\n') ;
tic
y1 = A'*x ;
toc
tm = toc ;

fprintf ('GrB AdotB:\n') ;
tic
y2 = GB_mex_AdotB (A,x) ;
toc

fprintf ('GrB A''xB auto select:\n') ;
tic
y3 = GB_mex_AxB (A,x, true) ;
tg = toc ;
fprintf ('GrB time is %g\n', tg) ;

fprintf ('GrB (A'')xB outer:\n') ;
tic
y3 = GB_mex_AxB (A',x) ;
toc

assert (isequal (y1, sparse (y0))) ;
assert (isequal (y1, y2)) ;
% assert (isequal (y1, y3)) ;
assert (norm (y1-y3,1) / norm (y1,1) < eps)

fprintf ('builtin: %10.4f  GB:auto: %10.4f speedup %10.4f\n', ...
    tm, tg, tm/tg) ;

%-------------------------------------------------------------------------------
fprintf ('\n--------------------------------------------------\n') ;

fprintf ('\nx''A where A is big and x is a dense vector\n') ;

fprintf ('builtin: x full:\n') ;
tic
y0 = z'*A ;
toc

fprintf ('builtin: x sparse:\n') ;
tic
y1 = x'*A ;
toc
tm = toc ;

fprintf ('GrB AdotB:\n') ;
tic
y2 = GB_mex_AdotB (x,A) ;
toc

fprintf ('GrB A''xB auto select:\n') ;
tic
y3 = GB_mex_AxB (x, A, true) ;
tg = toc ;

fprintf ('GrB (A''B outer:\n') ;
tic
y3 = GB_mex_AxB (x', A) ;
toc

assert (isequal (y1, sparse (y0))) ;
assert (isequal (y1, y2)) ;
% assert (isequal (y1, y3)) ;
assert (norm (y1-y2,1) / norm (y2,1) < eps)

fprintf ('builtin: %10.4f  GB:auto: %10.4f speedup %10.4f\n', ...
    tm, tg, tm/tg) ;

%-------------------------------------------------------------------------------
fprintf ('\n--------------------------------------------------\n') ;
fprintf ('\nA*x where A is big and x is a dense vector\n') ;

fprintf ('builtin: x full:\n') ;
tic
y0 = A*z ;
toc

fprintf ('builtin: x sparse:\n') ;
tic
y1 = A*x ;
toc
tm = toc ;

fprintf ('GrB AxB:\n') ;
tic
y3 = GB_mex_AxB (A, x, false) ;
tg = toc ;

assert (isequal (y1, sparse (y0))) ;
% assert (isequal (y1, y3)) ;
assert (norm (y1-y3,1) / norm (y1,1) < eps)

fprintf ('builtin: %10.4f  GB:auto: %10.4f speedup %10.4f\n', ...
    tm, tg, tm/tg) ;

%-------------------------------------------------------------------------------
fprintf ('\n--------------------------------------------------\n') ;

fprintf ('\nA''*x where A is big and x is a very sparse vector\n') ;
x = sprandn (n,1, 0.0001) ;

fprintf ('builtin: x sparse:\n') ;
tic
y1 = A'*x ;
toc
tm = toc ;

fprintf ('GrB AdotB:\n') ;
tic
y2 = GB_mex_AdotB (A,x) ;
toc

fprintf ('GrB A''xB auto select:\n') ;
tic
y3 = GB_mex_AxB (A,x, true) ;
tg = toc ;

fprintf ('GrB (A'')xB outer:\n') ;
tic
y3 = GB_mex_AxB (A',x) ;
toc

assert (isequal (y1, y2)) ;
% assert (isequal (y1, y3)) ;
assert (norm (y1-y3,1) / norm (y1,1) < eps)

fprintf ('builtin: %10.4f  GB:auto: %10.4f speedup %10.4f\n', ...
    tm, tg, tm/tg) ;

fprintf ('\ntest87: all tests passed\n') ;

nthreads_set (save, save_chunk) ;
