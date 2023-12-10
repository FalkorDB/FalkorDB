function test138
%TEST138 test assign, with coarse-only tasks in IxJ slice

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

[save_nthreads save_chunk] = nthreads_get ;
nthreads_set (2,1) ;

n = 1000 ;
k = 100 ;
C = sparse (rand (n)) ;

I = randperm (n,k) ;
J = randperm (n,k) ;
I0 = uint64 (I) - 1 ;
J0 = uint64 (J) - 1 ;

scalar = sparse (pi) ;

C0 = C ;
C0 (I,J) = scalar ;

C1 = GB_mex_assign  (C, [ ], [ ], scalar, I0, J0, [ ], 0) ;
C2 = GB_spec_assign (C, [ ], [ ], scalar, I , J , [ ], true) ;
GB_spec_compare (C1, C2) ;
assert (isequal (C0, C1.matrix)) ;

fprintf ('test138: all tests pass\n') ;
