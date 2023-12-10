function test42
%TEST42 test GrB_Matrix_build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n----------------------- performance tests for GrB_Matrix_build\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads_max = feature_numcores ;
% fprintf ('GraphBLAS: one thread\n') ;
% nthreads_set (1, chunk) ;

Prob = ssget ('HB/west0067')
A = Prob.A ;
[m n] = size (A) ;

[i j x] = find (Prob.A) ;
i = uint64 (i-1) ;
j = uint64 (j-1) ;
T = sparse (double (i+1), double (j+1), x) ;
assert (isequal (A,T)) ;

S = GB_mex_Matrix_build (i,j,x) ;
S = S.matrix ;
assert (isequal (A,S)) ;
assert (GB_spok (S) == 1) ;

nz = nnz (A) ;
p = randperm (nz) ;
i = i(p) ;
j = j(p) ;
x = x(p) ;

S = GB_mex_Matrix_build (i,j,x) ;
S = S.matrix ;
assert (isequal (A,(S')'))
assert (isequal (A,S)) ;
assert (GB_spok (S) == 1) ;

% duplicates
rng ('default') ;
i2 = floor (rand (100,1) * n) + 1 ;
j2 = floor (rand (100,1) * n) + 1 ;
x2 = rand (100,1) ;
i = [i ; uint64(i2-1)] ;
j = [j ; uint64(j2-1)] ;
x = [x ; x2] ;
T = sparse (double (i+1), double (j+1), x) ;

S = GB_mex_Matrix_build (i,j,x) ;
S = S.matrix ;
assert (isequal (spones (S), spones (T)))
assert (norm (S-T,1) == 0) ;
assert (GB_spok (T) == 1) ;

%-------------------------------------------------------------------------------
fprintf ('----------------------- matrix from collection, no sorting:\n') ;
Prob = ssget (939)
A = Prob.A ;
[m n] = size (A) ;
[i j x] = find (Prob.A) ;
i = uint64 (i-1) ;
j = uint64 (j-1) ;
i1 = double (i+1) ;
j1 = double (j+1) ;
fprintf ('built-in:\n') ;
tic
T = sparse (i1, j1, x) ;
toc
assert (isequal (A,T))

fprintf ('GrB:\n') ;
for nth = [1 2 4 8 16 20 40]
    if (nth > 2*nthreads_max)
        break ;
    end
    nthreads_set (nth) ;
    tic
    S = GB_mex_Matrix_build (i,j,x) ;
    S = S.matrix ;
    t = toc ;
    fprintf ('GrB with %d threads: %g\n', nth, t) ;
    assert (isequal (A,S))
    assert (GB_spok (S) == 1) ;
end

try 
    fprintf ('Csparse:\n') ;
    tic
    W = cs_sparse (i1,j1,x) ;
    toc
    ok = isequal (A,W) && (GB_spok (W) == 1) ;
catch
    % CSparse not available
    ok = true ;
end
assert (ok) ;

fprintf ('sparse2:\n') ;
try
    tic
    Y = sparse2 (i1,j1,x) ;
    toc
    ok = (isequal (A,Y)) && assert (GB_spok (Y) == 1) ;
catch
    % CHOLMOD not available
    ok = true ;
end
assert (ok) ;

%-------------------------------------------------------------------------------
fprintf ('\n----------------------- same matrix, but unsorted:\n') ;

rng ('default') ;
nz = length (x)
p = randperm (nz) ;
i1 = i1 (p) ;
j1 = j1 (p) ;
x  = x  (p) ;
i  = i  (p) ;
j  = j  (p) ;

fprintf ('built-in:\n') ;
tic
T = sparse (i1, j1, x) ;
toc

fprintf ('GrB:\n') ;
for nth = [1 2 4 8 16 20 40]
    if (nth > 2*nthreads_max)
        break ;
    end
    nthreads_set (nth) ;
    tic
    S = GB_mex_Matrix_build (i,j,x) ;
    S = S.matrix ;
    t = toc ;
    fprintf ('GrB with %d threads: %g\n', nth, t) ;
    assert (isequal (T,S))
    assert (GB_spok (S) == 1) ;
end

%-------------------------------------------------------------------------------
fprintf ('\n----------------------- random matrix, with duplicates:\n') ;
i2 = floor (rand (1000000,1) * n) + 1 ;
j2 = floor (rand (1000000,1) * n) + 1 ;
x2 = rand (1000000,1) ;
i = [i ; uint64(i2-1)] ;
j = [j ; uint64(j2-1)] ;
x = [x ; x2] ;
i1 = double (i+1) ;
j1 = double (j+1) ;

fprintf ('built-in:\n') ;
tic
T = sparse (i1, j1, x) ;
toc

fprintf ('GrB:\n') ;
for nth = [1 2 4 8 16 20 40]
    if (nth > 2*nthreads_max)
        break ;
    end
    nthreads_set (nth) ;
    tic
    S = GB_mex_Matrix_build (i,j,x) ;
    S = S.matrix ;
    % norm (T-S,1)
    t = toc ;
    fprintf ('GrB with %d threads: %g\n', nth, t) ;
    assert (isequal (T,S))
    assert (GB_spok (S) == 1) ;
end

try
    fprintf ('Csparse:\n') ;
    tic
    W = cs_sparse (i1,j1,x) ;
    toc
    ok = isequal (T,W) && (GB_spok (W) == 1) ;
catch
    % CSparse not available
    ok = true ;
end
assert (ok) ;

fprintf ('sparse2:\n') ;
try
    tic
    Y = sparse2 (i1,j1,x) ;
    toc
    % norm (T-Y,1)
    ok = (isequal (T,Y)) && assert (GB_spok (Y) == 1) ;
catch
    % CHOLMOD not available
    ok = true ;
end
assert (ok) ;

fprintf ('\n----------------------- same random matrix, but presorted:\n') ;
[ignore,p] = sortrows ([j i x]) ;
i = i (p) ;
j = j (p) ;
x = x (p) ;
i1 = i1 (p) ;
j1 = j1 (p) ;

fprintf ('built-in:\n') ;
tic
T = sparse (i1, j1, x) ;
toc

fprintf ('GrB:\n') ;
for nth = [1 2 4 8 16 20 40]
    if (nth > 2*nthreads_max)
        break ;
    end
    nthreads_set (nth) ;
    tic
    S = GB_mex_Matrix_build (i,j,x) ;
    S = S.matrix ;
    % norm (T-S,1)
    t = toc ;
    fprintf ('GrB with %d threads: %g\n', nth, t) ;
    assert (isequal (T,S))
    assert (GB_spok (S) == 1) ;
end

try
    fprintf ('CSparse:\n') ;
    tic
    W = cs_sparse (i1,j1,x) ;
    toc
    ok = isequal (T,W) && (GB_spok (W) == 1) ;
catch
    % CSparse not available
    ok = true ;
end
assert (ok) ;

fprintf ('sparse2:\n') ;
try
    tic
    Y = sparse2 (i1,j1,x) ;
    toc
    % norm (T-Y,1)
    ok = (isequal (T,Y)) && assert (GB_spok (Y) == 1) ;
catch
    % CHOLMOD not available
    ok = true ;
end
assert (ok) ;

fprintf ('\ntest42: all tests passed\n') ;

nthreads_set (save, save_chunk) ;
