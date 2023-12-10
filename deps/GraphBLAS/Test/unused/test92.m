function test92
%TEST92 test GB_subref: symbolic case

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test92:  test GB_subref: symbolic case\n') ;

rng ('default') ;
n = 100 ;
A = sprand (n, n, 0.1) ; 

% S = A (I,J) ;
% No column of A can be dense
% length(I) > 1
% I is not ":", not lo:hi
% sort not needed so I must be sorted
% 32*length(I) > entries in columns of A

I = [ 1 2 2 4:8 20:30 32 32 32 32] ;
I0 = uint64 (I) - 1 ;

J = 1:n ;
J0 = uint64 (J) - 1 ;

nI = length (I) ;
nJ = n ;

S = GB_mex_subref_symbolic (A, I0, J0) ;
[si,sj,sx] = find (S.matrix) ;

[i,j,x] = find (A) ;
nz = length (x) ;
x = 1:nz ;
X = sparse (i,j,x,n,n) ;

S2 = X (I,J) ;
[si2,sj2,sx2] = find (S2) ;

assert (isequal (si, si2)) ;
assert (isequal (sj, sj2)) ;
assert (isequal (sx+1, sx2)) ;

fprintf ('test92: all tests passed\n') ;

