function test66
%TEST66 test GrB_reduce

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest66: ---- quick test for GrB_reduce_to_scalar and vector\n') ;

rng ('default') ;
A = sparse (rand (4,3)) ;
x = full (sum (sum (A))) + 3.1416
c = GB_mex_reduce_to_scalar (3.1416, 'plus', 'plus', A) ;
assert (isequal (x,c))

tic
x = full (sum (sum (A))) ;
toc
tic
y = GB_mex_reduce_to_scalar (0, '', 'plus', A) ;
toc
assert (norm(x-y) < nnz (A) * eps * norm(x))

tic
x = full (sum (A (:))) ;
toc
tic
y = GB_mex_reduce_to_scalar (0, '', 'plus', A) ;
toc
assert (norm(x-y) < nnz (A) * eps * norm(x))
% assert (isequal (x,y))


% reduce to vector
y = sparse (4,1) ;
y = GB_mex_reduce_to_vector (y, [ ], '', 'plus', A) ;
y0 = sum (A')' ;
err = norm (y.matrix - y0) / norm (y0) ;
assert (err < 1e-14) ;
% assert (isequal (y.matrix, sum (A')'))

clear d
d.inp0 = 'tran' ;
y = sparse (3,1) ;
y = GB_mex_reduce_to_vector (y, [ ], '', 'plus', A, d) ;
y0 = sum (A)' ;
err = norm (y.matrix - y0) / norm (y0) ;
assert (err < 1e-14) ;
% assert (isequal (y.matrix, sum (A)'))

A = sprand (3e6, 3e6, 2e-6) ;
n = size (A,1) ;
yin = sparse (rand (n,1)) ;
fprintf ('\nbig matrix with %g million entries\n', nnz (A)/ 1e6) ;

% sum across the rows
fprintf ('row sum with accum:\n') ;
tic
y2 = yin + (sum (A,2)) ;
t1 = toc ;
tic
y = GB_mex_reduce_to_vector (yin, [ ], 'plus', 'plus', A) ;
t2 = toc ;
fprintf ('builtin: %g GraphBLAS %g speedup %g\n', t1, t2, t1/t2) ;
y1 = 1*y.matrix ;
err = norm (y1-y2,1) / norm (y2,1) ;
assert (err < 1e-14)
% assert (isequal (y.matrix, y2))

% sum across the rows (no accum)
fprintf ('row sum no accum:\n') ;
tic
y2 = (sum (A,2)) ;
t1 = toc ;
tic
y = GB_mex_reduce_to_vector (yin, [ ], [ ], 'plus', A) ;
t2 = toc ;
fprintf ('builtin: %g GraphBLAS %g speedup %g\n', t1, t2, t1/t2) ;
y1 = 1*y.matrix ;
err = norm (y1-y2,1) / norm (y2,1) ;
assert (err < 1e-14)
% assert (isequal (y.matrix, y2))

% sum down the columns
fprintf ('col sum with accum:\n') ;
yinrow = yin' ;
tic
y2 = yinrow + (sum (A,1)) ;
t1 = toc ;
tic
y = GB_mex_reduce_to_vector (yin, [ ], 'plus', 'plus', A, d) ;
t2 = toc ;
fprintf ('builtin: %g GraphBLAS %g speedup %g\n', t1, t2, t1/t2) ;
y1 = 1*y.matrix ;
err = norm (y1-y2',1) / norm (y2,1) ;
assert (err < 1e-14)
% assert (isequal (y.matrix, y2'))

% sum down the columns
fprintf ('col sum no accum:\n') ;
yempty = sparse (n,1) ;
tic
y2 = (sum (A,1)) ;
t1 = toc ;
tic
y = GB_mex_reduce_to_vector (yempty, [ ], [ ], 'plus', A, d) ;
t2 = toc ;
fprintf ('builtin: %g GraphBLAS %g speedup %g\n', t1, t2, t1/t2) ;
y1 = 1*y.matrix ;
err = norm (y1-y2',1) / norm (y2,1) ;
assert (err < 1e-14)
% assert (isequal (y.matrix, y2'))

% reduce to scalar
fprintf ('to scalar:\n') ;
tic
x = full (sum (sum (A))) ;
t1 = toc ;
tic
y = GB_mex_reduce_to_scalar (0, '', 'plus', A) ;
t2 = toc ;
fprintf ('builtin: %g GraphBLAS %g speedup %g\n', t1, t2, t1/t2) ;
assert (norm(x-y) < nnz (A) * eps * norm(x))

fprintf ('\ntest66: all tests passed\n') ;

