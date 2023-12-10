function test189
%TEST189 test large assignment

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test189 ----------- test large C(I,J)=A\n') ;

rng ('default') ;

n = 1e6 ;
nz = 1e6 ;
d = nz/n^2 ;
C = sprand (n, n, d) ;
A = sprand (n, n, d) ;
I = randperm (n) ;
J = randperm (n) ;
nthreads_set (4, 1) ;
% C (I,J) = A ;
I0 = uint64 (I) - 1 ;
J0 = uint64 (J) - 1 ;

C1 = GB_mex_assign (C, [ ], [ ], A, I0, J0, [ ], 0) ;
C2 = C ;
C2 (I, J) = A ;

assert (isequal (C1.matrix, C2)) ;

fprintf ('test189: all tests passed\n') ;
