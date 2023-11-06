function test95
%TEST95 performance test for GrB_transpose

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest95: performance tests : GrB_transpose \n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

rng ('default') ;
tol = 1e-12 ;

Prob = ssget (2662)
A = Prob.A ;
% A = sparse (rand (6000)) ;
[m n] = size (A) ;
Cin = sparse (n, m) ;
A (1,2) =1 ;

ntrials = 10 ;

tic
for trial = 1:ntrials
    C1 = A' ;
end
tmsum = toc ;
fprintf ('built-in  transpose time: %g\n', tmsum / ntrials) ;

% C = 0 ; C += A'
for trial = 1:ntrials
    tic
    C = GB_mex_transpose (Cin, [ ], 'plus', A) ;
    tg (trial) = toc ;
end
tgsum = sum (tg) ;
fprintf ('GraphBLAS transpose time: %g (for C=0 ; C+=A'')\n', tgsum / ntrials) ;
assert (isequal (C1, C.matrix)) ;
fprintf ('speedup over built-in: %g\n', tmsum / tgsum) ;

% C = A'
for trial = 1:ntrials
    tic
    C = GB_mex_transpose (Cin, [ ], [ ], A) ;
    tg (trial) = toc ;
end
tgsum = sum (tg) ;
fprintf ('GraphBLAS transpose time: %g (for C=A'')\n', tgsum / ntrials) ;
assert (isequal (C1, C.matrix)) ;
fprintf ('speedup over built-in: %g\n', tmsum / tgsum) ;

% sum across the rows
yin = sparse (rand (m,1)) ;
fprintf ('row sum (with accum):\n') ;
tic
y2 = yin + (sum (A,2)) ;
t1 = toc ;

toc
y = GB_mex_reduce_to_vector (yin, [ ], 'plus', 'plus', A) ;
t2 = tic ;
fprintf ('built-in: %g GraphBLAS %g speedup %g\n', t1, t2, t1/t2) ;
err = norm (1*(y.matrix) - y2, 1) ;
if (norm (y2) ~= 0)
    err = err / norm (y2) ;
end
assert (err < tol)
% assert (isequal (y.matrix, y2))

% sum across the rows, no accum
yin = sparse (rand (m,1)) ;
fprintf ('row sum (no accum):\n') ;
tic
y2 = (sum (A,2)) ;
t1 = toc ;

tic
y = GB_mex_reduce_to_vector (yin, [ ], [ ], 'plus', A) ;
t2 = toc ;
fprintf ('built-in: %g GraphBLAS %g speedup %g\n', t1, t2, t1/t2) ;
err = norm (1*(y.matrix) - y2, 1) ;
if (norm (y2) ~= 0)
    err = err / norm (y2) ;
end
assert (err < tol)
% assert (isequal (1*(y.matrix), y2))

% sum down the columns, no accum
yin = sparse (rand (m,1)) ;
fprintf ('col sum (no accum):\n') ;
tic
y2 = (sum (A,1)) ;
t1 = toc ;

desc.inp0 = 'tran' ;

tic
y = GB_mex_reduce_to_vector (yin, [ ], [ ], 'plus', A, desc) ;
t2 = toc ;
fprintf ('built-in: %g GraphBLAS %g speedup %g\n', t1, t2, t1/t2) ;
err = norm (1*(y.matrix) - y2', 1) ;
if (norm (y2) ~= 0)
    err = err / norm (y2) ;
end
assert (err < tol)
% assert (isequal (1*(y.matrix), y2'))

nthreads_set (save, save_chunk) ;
