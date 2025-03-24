function bench3(longtests)
%BENCH3 test and benchmark qsort and msort

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest44\n------------------------------------- qsort tests\n') ;

if (nargin < 1)
    longtests = 0 ;
end

nlist = [10 512 1024 2*1024 4*1024 8*1024 16*1024 32*1024 64*1024 103e3 200e3 1e6 ] ;
if (longtests)
    nlist = [nlist 10e6 100e6] ;
end
save = maxNumCompThreads ;

[save_nthreads save_chunk] = nthreads_get ;
fprintf ('maxNumCompThreads: %d  feature_numcores: %d\n', save, ...
    feature_numcores) ;
nthreads_max = 32 ;

rng ('default') ;

for n = nlist

fprintf ('\n\n\n\n========================== n %d (%g million)\n', ...
    n, n / 1e6) ;

fprintf ('\n----------------------- qsort 1b\n') ;
% qsort1b is not stable; it used only when I has unique values
I = uint64 (randperm (n))' ;
J = uint64 ((n/10)* rand (n,1)) ;
IJ = [I J] ;
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    maxNumCompThreads (nthreads) ;
    tic
    IJout = sortrows (IJ, 1) ;
    t = toc ;
    fprintf ('builtin: %2d threads: time %g\n', nthreads, t) ;
    if (nthreads == 1)
        t1 = t ;
    end
end
tic
[Iout, Jout] = GB_mex_qsort_1b (I, J) ;
t2 = toc ;
fprintf ('built-in: sortrows %g sec  qsort1b: %g speedup: %g\n', t1, t2, t1/t2);
assert (isequal ([Iout Jout], IJout))
clear Iout Jout IJout

clear tt
fprintf ('\n----------------------- qsort 1: 32 bit\n') ;
I = uint32 ((n/10)* rand (n,1)) ;
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    maxNumCompThreads (nthreads) ;
    tic
    IJout = sortrows (I) ;
    t = toc ;
    tt (nthreads) = t ;
    fprintf ('builtin: %2d threads: time %g\n', nthreads, t) ;
    if (nthreads == 1)
        t1 = t ;
    end
end
tic
Iout = GB_mex_qsort_1 (I) ;
t2 = toc ;
assert (isequal (Iout, IJout)) ;
clear Iout
fprintf ('built-in: sortrows %g sec  qsort1: %g speedup: %g\n', t1, t2, t1/t2) ;

fprintf ('\nmsort1: 32 bit\n') ;
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    tic
    [Iout] = GB_mex_msort_1 (I, nthreads) ;
    tp = toc ;
    if (nthreads == 1)
        tp1 = tp ;
    end
    assert (isequal ([Iout], IJout)) ;
    clear Iout
    fprintf ('msort1_32: %3d: %10.4g ', nthreads, tp) ;
    fprintf ('speedup vs 1: %8.3f ', tp1 / tp) ;
    fprintf ('speedup vs built-in: %8.3f\n', tt (nthreads) / tp) ;
end
clear IJout

fprintf ('\n----------------------- qsort 1: 64 bit\n') ;
I = uint64 (I) ;
clear tt
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    maxNumCompThreads (nthreads) ;
    tic
    IJout = sortrows (I) ;
    t = toc ;
    tt (nthreads) = t ;
    fprintf ('builtin: %2d threads: time %g\n', nthreads, t) ;
    if (nthreads == 1)
        t1 = t ;
    end
end
tic
Iout = GB_mex_qsort_1 (I) ;
t2 = toc ;
assert (isequal (Iout, IJout)) ;
clear Iout
fprintf ('built-in: sortrows %g sec  qsort1: %g speedup: %g\n', t1, t2, t1/t2) ;

fprintf ('\nmsort1: 64 bit\n') ;
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    tic
    [Iout] = GB_mex_msort_1 (I, nthreads) ;
    tp = toc ;
    if (nthreads == 1)
        tp1 = tp ;
    end
    assert (isequal ([Iout], IJout)) ;
    clear Iout
    fprintf ('msort1_64: %3d: %10.4g ', nthreads, tp) ;
    fprintf ('speedup vs 1: %8.3f ', tp1 / tp) ;
    fprintf ('speedup vs built-in: %8.3f\n', tt (nthreads) / tp) ;
end
clear IJout

if (n > 200e6)
    continue ;
end

fprintf ('\n----------------------- qsort 2: 32 bit\n') ;
I = uint32 ((n/10)* rand (n,1)) ;
J = uint32 (randperm (n))' ;
IJ = [I J] ;
clear tt
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    maxNumCompThreads (nthreads) ;
    tic
    IJout = sortrows (IJ) ;
    t = toc ;
    tt (nthreads) = t ;
    fprintf ('builtin: %2d threads: time %g\n', nthreads, t) ;
    if (nthreads == 1)
        t1 = t ;
    end
end
tic
[Iout, Jout] = GB_mex_qsort_2 (I, J) ;
t2 = toc ;
assert (isequal ([Iout Jout], IJout)) ;
clear Iout Jout 
fprintf ('built-in: sortrows %g sec  qsort2: %g speedup: %g\n', t1, t2, t1/t2) ;

fprintf ('\nmsort2: 32 bit\n') ;
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    tic
    [Iout, Jout] = GB_mex_msort_2 (I, J, nthreads) ;
    tp = toc ;
    if (nthreads == 1)
        tp1 = tp ;
    end
    assert (isequal ([Iout Jout], IJout)) ;
    clear Iout Jout
    fprintf ('msort2_32: %3d: %10.4g ', nthreads, tp) ;
    fprintf ('speedup vs 1: %8.3f ', tp1 / tp) ;
    fprintf ('speedup vs built-in: %8.3f\n', tt (nthreads) / tp) ;
end
clear IJout

fprintf ('\n----------------------- qsort 2: 64 bit\n') ;
I = uint64 (I) ;
J = uint64 (J) ;
IJ = [I J] ;
clear tt
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    maxNumCompThreads (nthreads) ;
    tic
    IJout = sortrows (IJ) ;
    t = toc ;
    fprintf ('builtin: %2d threads: time %g\n', nthreads, t) ;
    tt (nthreads) = t ;
    if (nthreads == 1)
        t1 = t ;
    end
end
tic
[Iout, Jout] = GB_mex_qsort_2 (I, J) ;
t2 = toc ;
assert (isequal ([Iout Jout], IJout)) ;
clear Iout Jout
fprintf ('built-in: sortrows %g sec  qsort2: %g speedup: %g\n', t1, t2, t1/t2) ;

fprintf ('\nmsort2: 64 bit\n') ;
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    tic
    [Iout, Jout] = GB_mex_msort_2 (I, J, nthreads) ;
    tp = toc ;
    if (nthreads == 1)
        tp1 = tp ;
    end
    assert (isequal ([Iout Jout], IJout)) ;
    clear Iout Jout
    fprintf ('msort2_64: %3d: %10.4g ', nthreads, tp) ;
    fprintf ('speedup vs 1: %8.3f ', tp1 / tp) ;
    fprintf ('speedup vs built-in: %8.3f\n', tt (nthreads) / tp) ;
end
clear IJout

fprintf ('\n----------------------- qsort 3: 32 bit\n') ;
I = uint32 ((n/10)* rand (n,1)) ;
J = uint32 ((n/10)* rand (n,1)) ;
K = uint32 (randperm (n))' ;
IJK = [I J K] ;
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    maxNumCompThreads (nthreads) ;
    tic
    IJKout = sortrows (IJK) ;
    t = toc ;
    fprintf ('builtin: %2d threads: time %g\n', nthreads, t) ;
    if (nthreads == 1)
        t1 = t ;
    end
end
for trials = 1:10
tic
[Iout, Jout, Kout] = GB_mex_qsort_3 (I, J, K) ;
t2 = toc ;
end
assert (isequal ([Iout Jout Kout], IJKout))
clear Iout Jout Kout
fprintf ('built-in: sortrows %g sec  qsort3: %g speedup: %g\n', t1, t2, t1/t2) ;
for trials = 1:10
tic
[Iout, Jout, Kout] = GB_mex_bsort (I, J, K) ;
t2 = toc ;
end
assert (isequal ([Iout Jout Kout], IJKout))
clear Iout Jout Kout
fprintf ('built-in: sortrows %g sec  bsort:  %g speedup: %g\n', t1, t2, t1/t2) ;

fprintf ('\nmsort3: 32\n') ;
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    tic
    [Iout, Jout, Kout] = GB_mex_msort_3 (I, J, K, nthreads) ;
    tp = toc ;
    if (nthreads == 1)
        tp1 = tp ;
    end
    assert (isequal ([Iout Jout Kout], IJKout)) ;
    clear Iout Jout Kout
    fprintf ('msort3_32: %3d: %10.4g ', nthreads, tp) ;
    fprintf ('speedup vs 1: %8.3f ', tp1 / tp) ;
    fprintf ('speedup vs built-in: %8.3f\n', tt (nthreads) / tp) ;
end

clear IJKout

fprintf ('\n----------------------- qsort 3: 64 bit\n') ;
I = uint64 (I) ;
J = uint64 (J) ;
K = uint64 (K) ;
IJK = [I J K] ;
clear tt
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    maxNumCompThreads (nthreads) ;
    tic
    IJKout = sortrows (IJK) ;
    t = toc ;
    fprintf ('builtin: %2d threads: time %g\n', nthreads, t) ;
    tt (nthreads) = t ;
    if (nthreads == 1)
        t1 = t ;
    end
end
for trials = 1:10
tic
[Iout, Jout, Kout] = GB_mex_qsort_3 (I, J, K) ;
t2 = toc ;
end
assert (isequal ([Iout Jout Kout], IJKout))
clear Iout Jout Kout
fprintf ('built-in: sortrows %g sec  qsort3: %g speedup: %g\n', t1, t2, t1/t2);
for trials = 1:10
tic
[Iout, Jout, Kout] = GB_mex_bsort (I, J, K) ;
t2 = toc ;
end
assert (isequal ([Iout Jout Kout], IJKout))
clear Iout Jout Kout
fprintf ('built-in: sortrows %g sec  bsort:  %g speedup: %g\n', t1, t2, t1/t2) ;

fprintf ('\nmsort3: 64\n') ;
for nthreads = [1 2 4 8 16 20 32 40 48 64 128 256]
    if (nthreads > nthreads_max)
        break ;
    end
    tic
    [Iout, Jout, Kout] = GB_mex_msort_3 (I, J, K, nthreads) ;
    tp = toc ;
    if (nthreads == 1)
        tp1 = tp ;
    end
    assert (isequal ([Iout Jout Kout], IJKout)) ;
    clear Iout Jout Kout
    fprintf ('msort3_64: %3d: %10.4g ', nthreads, tp) ;
    fprintf ('speedup vs 1: %8.3f ', tp1 / tp) ;
    fprintf ('speedup vs built-in: %8.3f\n', tt (nthreads) / tp) ;
end
clear IJKout

end

fprintf ('\ntest44: all tests passed\n') ;
nthreads_set (save_nthreads, save_chunk) ;

maxNumCompThreads (save) ;

