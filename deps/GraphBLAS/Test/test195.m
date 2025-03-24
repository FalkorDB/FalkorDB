function test195 (dohack)
%TEST195 test all variants of saxpy3

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test195 -------------- saxpy3 variants\n') ;

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    {1, 1, 1, 1, 1, 1,      1}, ... % (  2,   2)
    {1, 1, 4, 1, 1, 1,      1}, ... % (  2,   4)
    {1, 4, 1, 1, 1, 1,      1}, ... % (  3,   7)
    {2, 4, 1, 1, 1, 1,      1}, ... % (  3,  10)
    {4, 1, 1, 1, 1, 1,      1}, ... % (  1,  11)
    {4, 2, 1, 1, 1, 1,      1}, ... % (  2,  13)
    {4, 4, 1, 1, 1, 1,      1}, ... % (  1,  14)
    {1, 1, 1, 1, 2, 1,      1}, ... % (  2,  16)
    {1, 1, 4, 1, 2, 1,  65536}, ... % (  2,  18)
    {4, 2, 8, 1, 2, 3,      1}, ... % (  1,  19)
    {1, 8, 1, 1, 3, 1,      1}, ... % (  3,  22)
    {2, 8, 1, 1, 3, 1,      1}, ... % (  2,  24)
    {1, 4, 4, 2, 1, 1,      1}, ... % (  1,  25)
    {1, 1, 1, 2, 3, 1,      1}, ... % (  1,  26)
    {1, 8, 1, 2, 3, 1,      1}, ... % (  1,  27)
    {8, 1, 1, 3, 1, 1,      1}, ... % (  2,  29)
    {8, 2, 1, 3, 1, 1,      1}, ... % (  1,  30)
    {8, 4, 1, 3, 1, 1,      1}, ... % (  1,  31)
    } ;
end

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

rng ('default') ;
[nthreads_orig chunk_orig] = nthreads_get ;

% save current global settings, then modify them
save_hack = GB_mex_hack ;
hack = save_hack ;
if (nargin < 1)
    dohack = 2 ;
end
hack (1) = dohack ; GB_mex_hack (hack) ; % "very_costly" in saxpy3

k = 3 ;
n = 4 ;
m = 200 ;
desc.axb   = 'hash' ;
desc_s.axb = 'hash' ; desc_s.mask = 'structural' ;
dnot.axb   = 'hash' ; dnot.mask = 'complement' ;
dnot_s.axb = 'hash' ; dnot_s.mask = 'structural complement' ;

semiring.multiply = 'times' ;
semiring.add = 'plus' ;
semiring.class = 'double' ;

densities = [0.01 .1 inf] ;

% generate the test matrices

for ka = 1:3
    da = densities (ka) ;
    AA {ka} = GB_spec_random (m, k, da) ;
end

for kb = 1:3
    db = densities (kb) ;
    BB {kb} = GB_spec_random (k, n, db) ;
end

for km = 1:3
    dm = densities (km) ;
    M = GB_spec_random (m, n, dm) ;
    M.matrix = spones (M.matrix) ;
    MM {km} = M ;
end

% run the tests

for kk = 1:length(tasks)
    task = tasks {kk} ;
    asparsity = task {1} ;
    bsparsity = task {2} ;
    msparsity    = task {3} ;
    ka = task {4} ;
    kb = task {5} ;
    km = task {6} ;
    chunk = task {7} ;

    A = AA {ka} ;
    B = BB {kb} ;
    M = MM {km} ;
    A.sparsity = asparsity ;
    B.sparsity = bsparsity ;
    M.sparsity = msparsity ;

    nthreads_set (nthreads_orig, chunk_orig) ;

    % C = A*B
    C0 = A.matrix * B.matrix ;
    C1 = GB_spec_mxm (C0, [ ], [ ], semiring, A, B, desc) ;
    C2 = GB_mex_mxm  (C0, [ ], [ ], semiring, A, B, desc) ;
    GB_spec_compare (C1, C2, 0, 1e-12) ;
    err = norm (C0 - C2.matrix, 1) ;
    assert (err < 1e-12) ;

    nthreads_set (nthreads_orig, chunk) ;

    % C<M> = A*B
    C0 = (A.matrix * B.matrix) .* M.matrix ;
    C1 = GB_spec_mxm (C0, M, [ ], semiring, A, B, desc) ;
    C2 = GB_mex_mxm  (C0, M, [ ], semiring, A, B, desc) ;
    GB_spec_compare (C1, C2, 0, 1e-12) ;
    err = norm (C0 - C2.matrix, 1) ;
    assert (err < 1e-12) ;

    % C<!M> = A*B
    C0 = (A.matrix * B.matrix) .* (1 - M.matrix) ;
    C1 = GB_spec_mxm (C0, M, [ ], semiring, A, B, dnot) ;
    C2 = GB_mex_mxm  (C0, M, [ ], semiring, A, B, dnot) ;
    GB_spec_compare (C1, C2, 0, 1e-12) ;
    err = norm (C0 - C2.matrix, 1) ;
    assert (err < 1e-12) ;

    % C<M,struct> = A*B
    C0 = (A.matrix * B.matrix) .* M.matrix ;
    C1 = GB_spec_mxm (C0, M, [ ], semiring, A, B, desc_s) ;
    C2 = GB_mex_mxm  (C0, M, [ ], semiring, A, B, desc_s) ;
    GB_spec_compare (C1, C2, 0, 1e-12) ;
    err = norm (C0 - C2.matrix, 1) ;
    assert (err < 1e-12) ;

    % C<!M,struct> = A*B
    C0 = (A.matrix * B.matrix) .* (1 - M.matrix) ;
    C1 = GB_spec_mxm (C0, M, [ ], semiring, A, B, dnot_s) ;
    C2 = GB_mex_mxm  (C0, M, [ ], semiring, A, B, dnot_s) ;
    GB_spec_compare (C1, C2, 0, 1e-12) ;
    err = norm (C0 - C2.matrix, 1) ;
    assert (err < 1e-12) ;

    if (track_coverage)
        c = sum (GraphBLAS_grbcov > 0) ;
        d = c - clast ;
        if (d > 0)
            fprintf ('{%d, %d, %d, %d, %d, %d, %6d},', ...
                asparsity, bsparsity, msparsity, ka, kb, km, chunk) ;
            fprintf (' ... %% (%3d, %3d)\n', d, c-cfirst) ;
        end
        clast = c ;
    else
        fprintf ('.') ;
    end
end

% restore global settings
GB_mex_hack (save_hack) ;
nthreads_set (nthreads_orig, chunk_orig) ;

fprintf ('\ntest195: all tests passed\n') ;

