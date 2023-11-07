function test30b
%TEST30B performance test GB_mex_assign, scalar expansion

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[save_nthreads save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

Prob = ssget (2662) ;
A = Prob.A ;

[m n] = size (A) ;

ni =  500 ;
nj = 1000 ;
I = randperm (m,ni) ;
J = randperm (n,nj) ;
I0 = uint64 (I-1) ;
J0 = uint64 (J-1) ;

scalar = sparse (pi) ;

% tic/toc includes the mexFunction overhead of making a deep copy
% of the input matrix.

fprintf ('start GraphBLAS:\n') ;
tic
C2 = GB_mex_assign (A, [], [], scalar, I0, J0, []) ;
t = toc ;

C = A ; 
fprintf ('start built-in method:\n') ;
tic 
C (I,J) = scalar ;
tm = toc

fprintf ('GraphBLAS speedup over built-in method: %g\n',  tm/t) ;

assert (isequal (C, C2.matrix)) ;
fprintf ('\ntest30b: all tests passed\n') ;

nthreads_set (save_nthreads, save_chunk) ;

