function test107
%TEST107 user-defined terminal monoid

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test107: reduce with built-in and  user-defined terminal monoids\n') ;

rng ('default') ;

save = nthreads_get ;
if (nargin < 1)
    fulltest = false ;
end
nthreads_max = GB_mex_omp_max_threads ;
if (fulltest)
    nthreads_list = [1 2 4 8 16 20 40 64 160] ;
else
    nthreads_list = [1 nthreads_max 2*nthreads_max] ;
end

% create a matrix with entries [0..2]
A = 2 * sparse (rand (4)) ;
s = full (max (max (A))) ;

c = GB_mex_reduce_terminal (A, 2) ;
assert (c == s) ;

% now add the terminal value somewhere
A (1,2) = 2 ;
s = full (max (max (A))) ;
c = GB_mex_reduce_terminal (A, 2) ;
assert (c == s) ;
clear A

ntrials = 10 ;

%-------------------------------------------------------------------------------
% big matrix ...
fprintf ('\nbig matrix, no early exit\n') ;
if (fulltest)
    n = 6000 ;
else
    n = 1000 ;
end
A = sparse (rand (n)) ;

tic
for trial = 1:ntrials
    s = full (max (max (A))) ;
end
tm = toc ;
fprintf ('builtin max: %g\n', tm) ;
for nthreads = nthreads_list
    fprintf ('\n') ;
    if (nthreads > 2*nthreads_max)
        break ;
    end
    nthreads_set (nthreads) ;
    tic
    for trial = 1:ntrials
        c1 = GB_mex_reduce_to_scalar (0, [ ], 'max', A) ;
    end
    tg = toc ;
    assert (s == c1) ;
    if (nthreads == 1)
        t1 = tg ;
    end

    fprintf ('nthreads %3d built-in      %g  speedup %g\n', nthreads, tg, t1/tg) ;

    tic
    for trial = 1:ntrials
        c2 = GB_mex_reduce_terminal (A, 1) ;    % user-defined
    end
    t2 = toc ;
    fprintf ('nthreads %3d %g\n', nthreads, t2) ;
    assert (s == c2) ;

    tic
    for trial = 1:ntrials
        c3 = GB_mex_reduce_terminal (A, 2) ;    % user-defined
    end
    t3 = toc ;
    fprintf ('nthreads %3d %g\n', nthreads, t3) ;
    assert (s == c3) ;

end

%-------------------------------------------------------------------------------
fprintf ('\nbig matrix, with early exit\n') ;

A (n,1) = 1 ;

tic
for trial = 1:ntrials
    s = full (max (max (A))) ;
end
tm = toc ;
fprintf ('builtin max: %g\n', tm) ;
for nthreads = nthreads_list
    fprintf ('\n') ;
    if (nthreads > nthreads_max)
        break ;
    end
    nthreads_set (nthreads) ;
    tic
    for trial = 1:ntrials
        c1 = GB_mex_reduce_to_scalar (0, [ ], 'max', A) ;
    end
    t1 = toc ;
    fprintf ('nthreads %3d built-in      %g\n', nthreads, t1) ;
    tic
    for trial = 1:ntrials
        c2 = GB_mex_reduce_terminal (A, 1) ;    % user-defined
    end
    t2 = toc ;
    fprintf ('nthreads %3d %g\n', nthreads, t2) ;
    assert (s == c1) ;
    assert (s == c2) ;
end

%-------------------------------------------------------------------------------
fprintf ('\nbig matrix, with inf \n') ;

A (n,1) = inf ;

tic
for trial = 1:ntrials
    s = full (max (max (A))) ;
end
tm = toc ;
fprintf ('builtin max: %g\n', tm) ;
for nthreads = nthreads_list
    fprintf ('\n') ;
    if (nthreads > nthreads_max)
        break ;
    end
    nthreads_set (nthreads) ;
    tic
    for trial = 1:ntrials
        c1 = GB_mex_reduce_to_scalar (0, [ ], 'max', A) ;
    end
    t1 = toc ;
    fprintf ('nthreads %3d built-in      %g\n', nthreads, t1) ;
    tic
    for trial = 1:ntrials
        c2 = GB_mex_reduce_terminal (A, inf) ;
    end
    t2 = toc ;
    fprintf ('nthreads %3d %g\n', nthreads, t2) ;
    assert (s == c1) ;
    assert (s == c2) ;
end

%-------------------------------------------------------------------------------
fprintf ('\nbig matrix, with 2 \n') ;

A (n,1) = 2 ;

tic
for trial = 1:ntrials
    s = full (max (max (A))) ;
end
tm = toc ;
fprintf ('builtin max: %g\n', tm) ;
for nthreads = nthreads_list
    fprintf ('\n') ;
    if (nthreads > nthreads_max)
        break ;
    end
    nthreads_set (nthreads) ;
    tic
    for trial = 1:ntrials
        c1 = GB_mex_reduce_to_scalar (0, [ ], 'max', A) ;
    end
    t1 = toc ;
    fprintf ('nthreads %3d built-in      %g\n', nthreads, t1) ;
    tic
    for trial = 1:ntrials
        c2 = GB_mex_reduce_terminal (A, 2) ;
    end
    t2 = toc ;
    fprintf ('nthreads %3d %g\n', nthreads, t2) ;
    assert (s == c1) ;
    assert (s == c2) ;
end

%-------------------------------------------------------------------------------
fprintf ('\nbig matrix, with nan\n') ;

A (n,1) = nan ;

tic
for trial = 1:ntrials
    s = full (max (max (A))) ;
end
tm = toc ;
fprintf ('builtin max: %g\n', tm) ;
for nthreads = nthreads_list
    fprintf ('\n') ;
    if (nthreads > nthreads_max)
        break ;
    end
    nthreads_set (nthreads) ;
    tic
    for trial = 1:ntrials
        c1 = GB_mex_reduce_to_scalar (0, [ ], 'max', A) ;
    end
    t1 = toc ;
    fprintf ('nthreads %3d built-in      %g\n', nthreads, t1) ;
    assert (s == c1) ;
end

assert (s == c1) ;

%-------------------------------------------------------------------------------
fprintf ('\nsum\n') ;

A (n,1) = 1 ;

tic
for trial = 1:ntrials
    ss = full (sum (sum (A))) ;
end
tm = toc ;
fprintf ('builtin sum: %g\n', tm) ;
for nthreads = nthreads_list
    fprintf ('\n') ;
    if (nthreads > nthreads_max)
        break ;
    end
    nthreads_set (nthreads) ;
    tic
    for trial = 1:ntrials
        cc = GB_mex_reduce_to_scalar (0, [ ], 'plus', A) ;
    end
    t1 = toc ;
    fprintf ('nthreads %3d built-in      %g\n', nthreads, t1) ;
    assert (s == c1) ;
end

err = abs (ss - cc) / ss 
assert (err < 1e-12) ;

nthreads_set (save) ;
fprintf ('test107: all tests passed\n') ;

