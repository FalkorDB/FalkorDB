function test44(longtests)
%TEST44 test qsort

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest44\n------------------------------------- qsort tests\n') ;

if (nargin < 1)
    longtests = 0 ;
end

nlist = [0 1 5 100 50e3 103e3 200e3 1e6 ] ;
if (longtests)
    nlist = [nlist 10e6 100e6] ;
else
end

[save_nthreads save_chunk] = nthreads_get ;
nthreads_max = feature_numcores ;

rng ('default') ;

for n = nlist

fprintf ('\n========================== n %g million\n', n / 1e6) ;

fprintf ('\n----------------------- qsort 1b\n') ;

% qsort1b is not stable; it used only when I has unique values
I = int64 (randperm (n))' ;
J = int64 ((n/10)* rand (n,1)) ;
IJ = [I J] ;

tic
IJout = sortrows (IJ, 1) ;
t = toc ;

tic
[Iout, Jout] = GB_mex_qsort_1b (I, J) ;
t2 = toc ;

fprintf ('built-in: sortrows %g sec  qsort1b: %g  speedup: %g\n', t, t2, t/t2) ;

assert (isequal ([Iout Jout], IJout))

fprintf ('\n----------------------- qsort 2\n') ;

I = int64 ((n/10)* rand (n,1)) ;
J = int64 (randperm (n))' ;
IJ = [I J] ;

tic
IJout = sortrows (IJ) ;
t = toc ;

tic
[Iout, Jout] = GB_mex_qsort_2 (I, J) ;
t2_just = toc ;
t2 = toc ;
assert (isequal ([Iout Jout], IJout));

fprintf ('built-in: sortrows %g sec  qsort2: %g %g speedup: %g\n', ...
    t, t2, t2_just, t/t2) ;

for nthreads = [1 2 4 8 16 20 32 40 64 128 256]
    if (nthreads > 2*nthreads_max)
        break ;
    end
    tic
    [Iout, Jout] = GB_mex_msort_2 (I, J, nthreads) ;
    tp = toc ;
    if (nthreads == 1)
        tp1 = tp ;
    end
    assert (isequal ([Iout Jout], IJout));
    fprintf ('msort2: %3d: %10.4g ', nthreads, tp) ;
    fprintf ('speedup vs 1: %8.3f ', tp1 / tp) ;
    fprintf ('speedup vs built-in: %8.3f\n', t / tp) ;
end

fprintf ('\n----------------------- qsort 3\n') ;

I = int64 ((n/10)* rand (n,1)) ;
J = int64 ((n/10)* rand (n,1)) ;
K = int64 (randperm (n))' ;
IJK = [I J K] ;

tic
IJKout = sortrows (IJK) ;
t = toc ;

tic
[Iout, Jout, Kout] = GB_mex_qsort_3 (I, J, K) ;
t2_just = toc ;
t2 = toc ;
assert (isequal ([Iout Jout Kout], IJKout))

fprintf ('built-in: sortrows %g sec  qsort3: %g  speedup: %g\n', t, t2, t/t2) ;

for nthreads = [1 2 4 8 16 20 32 40 64 128 256]
    if (nthreads > 2*nthreads_max)
        break ;
    end
    tic
    [Iout, Jout, Kout] = GB_mex_msort_3 (I, J, K, nthreads) ;
    tp = toc ;
    if (nthreads == 1)
        tp1 = tp ;
    end
    assert (isequal ([Iout Jout Kout], IJKout));
    fprintf ('msort3: %3d: %10.4g ', nthreads, tp) ;
    fprintf ('speedup vs 1: %8.3f ', tp1 / tp) ;
    fprintf ('speedup vs built-in: %8.3f\n', t / tp) ;
end

end

fprintf ('\ntest44: all tests passed\n') ;
nthreads_set (save_nthreads, save_chunk) ;

