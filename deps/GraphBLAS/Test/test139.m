function test139
%TEST139 merge sort, special cases

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test139 --------------- merge sort, special cases\n') ;
rng ('default') ;

n = 1e6 ;
I = 42 * ones (n,1) ;
J = (1:n)' ;
K = 100 * ones (n,1) ;

I0 = int64 (I) ;
J0 = int64 (J) ;
K0 = int64 (K) ;

IJ1 = sortrows ([I0 J0]) ;
[a b] = GB_mex_msort_2 (I0, J0, 2) ;
assert (isequal (IJ1, [a b])) ;

IJ3 = sortrows ([I0 J0 K0]) ;
[a b c] = GB_mex_msort_3 (I0, J0, K0, 2) ;
assert (isequal (IJ3, [a b c])) ;

IJ1 = sortrows ([J0 I0]) ;
[a b] = GB_mex_msort_2 (J0, I0, 2) ;
assert (isequal (IJ1, [a b])) ;

fprintf ('test139 --------------- all tests passed\n') ;
