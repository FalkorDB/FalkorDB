function test91
%TEST91 test subref performance on dense vectors

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n------------------------------ testing GB_mex_Matrix_subref\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

ntrials = 10 ;

% addpath old
rng ('default')

n = 10 * 1e6 ;
A = sparse (rand (n,1)) ;
fprintf ('A is a sparse %d-by-1 vector, all nonzero\n', n) ;
F = full (A) ;


for ilen = [1 10 100 1000 10000 100000 1e6]

    fprintf ('\n----- C(I) = A (I), I is random with length(I) = %d\n', ilen) ;

    I  = irand (1, n, ilen, 1) ;
    I0 = uint64 (I-1) ;

    tic
    for trials = 1:ntrials
        C1 = A (I) ;
    end
    tm = toc ;
    fprintf ('built-in sparse: %g sec\n', tm) ;

    tic
    for trials = 1:ntrials
        Cfull = F (I) ;
    end
    tf = toc ;
    fprintf ('built-in full:   %g sec\n', tf) ;

    J0 = uint64 (0) ;

    tic
    for trials = 1:ntrials
        C3 = GB_mex_Matrix_subref (A, I0, J0) ;
    end
    tg = toc ;
    fprintf ('GraphBLAS:     %g sec speedup %g\n', tg, tm/tg) ;
    assert (isequal (C1, C3)) ;

end

fprintf ('\n----- C(:) = A (:)\n') ;

tic
for trials = 1:ntrials
    C1 = A (:) ;
end
tm = toc ;
fprintf ('built-in:        %g\n', tm) ;

tic
for trials = 1:ntrials
    C3 = GB_mex_Matrix_subref (A, [ ], J0) ;
end
tg = toc ;
assert (isequal (C1, C3)) ;
fprintf ('GraphBLAS:     %g sec speedup %g\n', tg, tm/tg) ;

F = full (A) ;

tic
for trials = 1:ntrials
    C0 = F (:) ;
    C0 (1) = 1 ;    % make sure the copy gets done, not a lazy copy
end
tf = toc ;
fprintf ('\nbuilt-in (full): %g\n', tf) ;

nthreads_set (save, save_chunk) ;
