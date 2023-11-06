function test135
%TEST135 reduce-to-scalar, built-in monoids with terminal values

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test135: reduce to scalar\n') ;

rng ('default') ;

n = 10e6 ;

[save_nthreads, save_chunk] = nthreads_get ;
nthreads_max = feature_numcores

%-------------------------------------------------------------------------------
fprintf ('================== int8 min:\n') ;

X = rand (n,1) ;
X = (256 * X) - 128 ;
X = int8 (X) ;
s = int8 (inf) ;
tic
c0 = min (X) ;
tm = toc ;
fprintf ('built-in: %g sec\n', tm) ;

A.matrix = sparse (double (X)) ;
A.pattern = logical (spones (X)) ;
A.class = 'int8' ;

nthreads_set (1,1) ;
tic
c1 = GB_mex_reduce_to_scalar (s, [ ], 'min', A) ;
t1 = toc ;
assert (c1 == c0) ;
fprintf ('1 thread  %g sec\n', t1) ;

nthreads_set (nthreads_max,1) ;
tic
c2 = GB_mex_reduce_to_scalar (s, [ ], 'min', A) ;
t2 = toc ;
assert (c2 == c0) ;
fprintf ('%d threads %g sec\n', nthreads_max, t2) ;

%-------------------------------------------------------------------------------
fprintf ('================== double min:\n') ;

X = sparse (rand (n,1)) ;

A.matrix = X ;
A.pattern = logical (spones (X)) ;
A.class = 'double' ;

tic
c0 = min (X) ;
tm = toc ;
fprintf ('built-in: %g sec\n', tm) ;

s = double (inf) ;

nthreads_set (1,1) ;
tic
c1 = GB_mex_reduce_to_scalar (s, [ ], 'min', A) ;
t1 = toc ;
assert (c1 == c0) ;
fprintf ('1 thread  %g sec\n', t1) ;

nthreads_set (nthreads_max,1) ;
tic
c2 = GB_mex_reduce_to_scalar (s, [ ], 'min', A) ;
t2 = toc ;
assert (c2 == c0) ;
fprintf ('%d threads %g sec\n', nthreads_max, t2) ;

%-------------------------------------------------------------------------------
fprintf ('================== double sum:\n') ;

X = rand (n,1) ;
tic
c0 = sum (X) ;
tm = toc ;
fprintf ('built-in: %g sec (full)\n', tm) ;

X = sparse (X) ;

A.matrix = X ;
A.pattern = logical (spones (X)) ;
A.class = 'double' ;

tic
c0 = full (sum (X)) ;
tm = toc ;
fprintf ('built-in: %g sec (sparse)\n', tm) ;

s = double (inf) ;

nthreads_set (1,1) ;
tic
c1 = GB_mex_reduce_to_scalar (s, [ ], 'plus', A) ;
t1 = toc ;
assert (norm (c1 - c0) / norm (c0) < 1e-12) ;
fprintf ('1 thread  %g sec\n', t1) ;

nthreads_set (nthreads_max,1) ;
tic
c2 = GB_mex_reduce_to_scalar (s, [ ], 'plus', A) ;
t2 = toc ;
assert (norm (c2 - c0) / norm (c0) < 1e-12) ;
fprintf ('%d threads %g sec\n', nthreads_max, t2) ;

%-------------------------------------------------------------------------------
nthreads_set (save_nthreads, save_chunk) ;
fprintf ('test135: all tests passed\n') ;
