function test294
%TEST294 reduce with zombies

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
fprintf ('test294 --------------- reduce with zombies\n') ;

n = 10000 ;
d = 0.01 ;
A = sprand (n, n, d) ;
[i j x] = find (A) ;
z = x (2:2:end) ;
s1 = sum (z) ;

[nth_save chunk_save] = nthreads_get ;

for nthreads = [1 8]
    nthreads_set (nthreads, 1024) ;
    s2 = GB_mex_reduce_with_zombies (A) ;
    err = abs (s1 - s2) / abs (s1) ;
    assert (err < 1e-10) ;
end

nthreads_set (nth_save, chunk_save) ;
fprintf ('\ntest294: all tests passed\n') ;

