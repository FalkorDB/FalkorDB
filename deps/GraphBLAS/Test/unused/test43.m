function test43
%TEST43 test subref

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n------------------------------ testing GB_mex_Matrix_subref\n') ;

rng ('default')

A = sparse (rand (3,4)) ;
I = uint64 (0:2) ;
J = uint64 (0:3) ;

C0 = A (I+1,J+1) ;
C = GB_mex_Matrix_subref (A, I, J) ;
assert (isequal (C, C0))

C0 = A (:,:) ;
C = GB_mex_Matrix_subref (A, [ ], [ ]) ;
assert (isequal (C, C0))

C0 = A (1,:) ;
C = GB_mex_Matrix_subref (A, uint64(0), [ ]) ;
assert (isequal (C, C0))

C0 = A (:,1) ;
C = GB_mex_Matrix_subref (A, [ ], uint64(0)) ;
assert (isequal (C, C0))

I = uint64 ([0 1]) ;
J = uint64 ([0 1]) ;
A = sparse (rand (4)) ;
C0 = full (A (I+1,J+1)) ;
full (A) ;
C = GB_mex_Matrix_subref (A, I, J) ;
full (C) ;
assert (isequal (C, C0))

I = uint64 ([2 1]) ;
J = uint64 ([3 1]) ;
C0 = full (A (I+1,J+1)) ;
C = GB_mex_Matrix_subref (A, I, J) ;
C = full (C) ;
assert (isequal (C, C0)) ;

fprintf ('-------------------------- problem:\n') ;
% Prob = ssget ('HB/west0067') ; A = Prob.A ;
  Prob = ssget (939) ; A = Prob.A ;
% A = sparse (rand (8000)) ;
A(1,2) =44 ;

fprintf ('-------------------------- case 5, ni large, qsort, no dupl:\n') ;
p = amd (A) ;
fprintf ('builtin:\n') ;
tic
A1 = A (p,p) ;
t0 = toc ;
p0 = uint64 (p-1) ;
fprintf ('GB:\n') ;
tic
A2 = GB_mex_Matrix_subref (A, p0, p0) ;
t1 = toc ;

fprintf ('CSparse permute:\n') ;
try
    tic
    A3 = cs_permute (A, p, p) ;
    t2 = toc ;
    ok = isequal (A1, A3) ;
catch
    % CSparse not available
    t2 = 0 ;
    ok = true ;
end
assert (ok) ;

assert (isequal (A1, A2)) ;
fprintf ('builtin %g GrB %g CSparse %g\n', t0, t1, t2) ;

n = size (A,1) ;

fprintf ('-------------------------- case 5, ni large, qsort, no dupl:\n') ;
I = uint64 (randperm (floor (n/2))) ;
J = uint64 (randperm (floor (n/2))) ;
I1 = I + 1 ;
J1 = J + 1 ;
fprintf ('builtin:\n') ;
    tic
    C0 = A (I1,J1) ;
    toc
fprintf ('GB:\n') ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    toc
    assert (isequal (C0, C1)) ;

fprintf ('-------------------------- contig:\n') ;
I = sort (I) ;
J = sort (J) ;
I1 = I + 1 ;
J1 = J + 1 ;
fprintf ('length (I), %d min %d max %d\n', length (I), min (I), max (I)) ;
fprintf ('builtin:\n') ;
    tic
    C0 = A (I1,J1) ;
    toc
fprintf ('GB:\n') ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    toc
    assert (isequal (C0, C1)) ;

fprintf ('-------------------------- case 5, ni large, qsort, with dupl:\n') ;
I = uint64 (floor (n * rand (n,1))) ;
J = uint64 (floor (n * rand (n,1))) ;
I1 = I + 1 ;
J1 = J + 1 ;
fprintf ('builtin:\n') ;
    tic
    C0 = A (I1,J1) ;
    toc
fprintf ('GB:\n') ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    toc
    assert (isequal (C0, C1)) ;

    fprintf ('double transpose time for built-in method:\n') ;
    tic
    C0 = C0'' ;
    toc

fprintf ('-------------------------- first rows:\n') ;
i = uint64 (floor(n/2)) ;
I = [ ] ;
J = uint64 (0:n-1) ;
J1 = J + 1 ;
for k = i:(i+50)
    if (k >= size (A,1))
        break ;
    end
    I = [I k] ;
    I1 = I+1 ;
    fprintf (' %3d rows: ', length (I)) ;
    tic
    C0 = A (I1,:) ;
    t0 = toc ;
    fprintf ('C0: %9d %9d  ', nnz (C0), nzmax (C0)) ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, [ ]) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin: %0.6f  speedup %6.2f\n', t0, t0/t1) ;
end

fprintf ('-------------------------- last row:\n') ;
i = uint64 (n-1) ;
I = i ;
I1 = I+1 ;
tic
C0 = A (I1,:) ;
t0 = toc ;
fprintf ('C0: %9d %9d  ', nnz (C0), nzmax (C0)) ;
tic
C1 = GB_mex_Matrix_subref (A, I, [ ]) ;
t1 = toc ;
assert (isequal (C0, C1)) ;
fprintf ('builtin: %0.6f  speedup %6.2f\n', t0, t0/t1) ;

fprintf ('-------------------------- contig, sorted:\n') ;
I = uint64 (0:floor(n/2)) ;
J = uint64 (0:floor(n/2)) ;
I1 = I + 1 ;
J1 = J + 1 ;
fprintf ('builtin:\n') ;
    tic
    C0 = A (I1,J1) ;
    toc
fprintf ('GB:\n') ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    toc
    assert (isequal (C0, C1)) ;

fprintf ('-------------------------- contig lower half, sorted:\n') ;
I = uint64 (floor(n/2):n-1) ;
J = uint64 (floor(n/2):n-1) ;
I1 = I + 1 ;
J1 = J + 1 ;
fprintf ('length (I), %d min %d max %d\n', length (I), min (I), max (I)) ;
fprintf ('builtin:\n') ;
    tic
    C0 = A (I1,J1) ;
    toc
fprintf ('GB:\n') ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    toc
    assert (isequal (C0, C1)) ;

fprintf ('-------------------------- one row:\n') ;
for i = 0:1000:n-1
    I = uint64 (i) ;
    J = uint64 (0:n-1) ;
    I1 = I+1 ;
    J1 = J+1 ;
    tic
    C0 = A (I1,J1) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin: %0.6f  speedup %6.2f\n', t0, t0/t1) ;
end



fprintf ('-------------------------- two rows:\n') ;
for i = 0:1000:n-2
    I = uint64([i i+1]) ;
    J = uint64 (0:n-1) ;
    I1 = I+1 ;
    J1 = J+1 ;
    tic
    C0 = A (I1,J1) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin: %0.6f  speedup %6.2f\n', t0, t0/t1) ;
end


fprintf ('-------------------------- three contig rows:\n') ;
for i = 0:1000:n-3
    I = uint64([i i+1 i+2]) ;
    J = uint64 (0:n-1) ;
    I1 = I+1 ;
    J1 = J+1 ;
    tic
    C0 = A (I1,J1) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin: %0.6f  speedup %6.2f\n', t0, t0/t1) ;
end

fprintf ('-------------------------- four contig rows:\n') ;
for i = 0:1000:n-4
    I = uint64([i i+1 i+2 i+3]) ;
    J = uint64 (0:n-1) ;
    I1 = I+1 ;
    J1 = J+1 ;
    tic
    C0 = A (I1,J1) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin: %0.6f  speedup %6.2f\n', t0, t0/t1) ;
end


fprintf ('-------------------------- one column :\n') ;
for j = 0:1000:n-4
    % I = uint64(0:n-1) ;
    J = uint64 (j) ;
    % I1 = I+1 ;
    J1 = J+1 ;
    tic
    C0 = A (:,J1) ;
    t0 = toc ;
    tic
    C1 = GB_mex_Matrix_subref (A, [ ], J) ;
    t1 = toc ;
    assert (isequal (C0, C1)) ;
    fprintf ('builtin: %0.6f  speedup %6.2f\n', t0, t0/t1) ;
end


fprintf ('-------------------------- A (:,2:n):\n') ;
I = [ ] ;
J = uint64 (1:n-1) ;
I1 = I + 1 ;
J1 = J + 1 ;
fprintf ('builtin:\n') ;
    tic
    C0 = A (:,J1) ;
    toc
fprintf ('GB:\n') ;
    tic
    C1 = GB_mex_Matrix_subref (A, [ ], J) ;
    toc
    assert (isequal (C0, C1)) ;

fprintf ('-------------------------- case 6: ni large, no qsort, dupl :\n') ;
I = uint64 (floor (n/2 : 0.5 : n-1)) ;
J = uint64 (1:n-1) ;
I1 = I + 1 ;
J1 = J + 1 ;
fprintf ('builtin:\n') ;
    tic
    C0 = A (I1,J1) ;
    toc
fprintf ('GB:\n') ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    toc
    assert (isequal (C0, C1)) ;

fprintf ('-------------------------- case 7: ni large, no qsort, no dupl :\n') ;
I = uint64 ([ floor(n/2) floor((2+n/2):n-1) ]) ;
J = uint64 (1:n-1) ;
I1 = I + 1 ;
J1 = J + 1 ;
fprintf ('builtin:\n') ;
    tic
    C0 = A (I1,J1) ;
    toc
fprintf ('GB:\n') ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    toc
    assert (isequal (C0, C1)) ;

fprintf ('-------------------------- case 3: contig :\n') ;
I = uint64 ([ floor(n/2:n-1) ]) ;
J = uint64 (1:n-1) ;
I1 = I + 1 ;
J1 = J + 1 ;
fprintf ('builtin:\n') ;
    tic
    C0 = A (I1,J1) ;
    toc
fprintf ('GB:\n') ;
    tic
    C1 = GB_mex_Matrix_subref (A, I, J) ;
    toc
    assert (isequal (C0, C1)) ;

fprintf ('\ntest43: all tests passed\n') ;

