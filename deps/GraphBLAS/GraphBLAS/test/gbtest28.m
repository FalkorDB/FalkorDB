function gbtest28 (ghb)
%GBTEST28 test [GrB,GhB].build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

fprintf ('\ngbtest28: testing %s.build and compare with A=sparse(i,j,x)\n', ...
    gtb_name) ;
nthreads = gtb_threads (ghb) ;
fprintf ('using %d threads in GraphBLAS\n', nthreads) ;

m = 10 ;
n = 5 ;
A = sprand (m, n, 0.5) ;

[i, j, x] = find (A) ;

C = gtb_build (ghb, i, j, x, m, n) ;

assert (gbtest_eq (C, A)) ;

fprintf ('\nGenerating large test matrix; please wait ...\n') ;
n = 1000 ;
nz = 7000 ;
density = nz / n^2 ;
tic
A = sprand (n, n, density) ;
B = sprand (n, n, density) ;
A = kron (A,B) ;
clear B
t = toc ;
n = size (A, 1) ;
fprintf ('%12.4f sec : A n-by-n, with n: %g nnz: %g\n', t, n, nnz (A)) ;

[i, j, x] = find (A) ;
[m, n] = size (A) ;

i0 = uint64 (i) - 1 ;
j0 = uint64 (j) - 1 ;

d.kind = 'sparse' ;
desc0.base = 'zero-based' ;

fprintf ('\nwith [I J] already sorted on input:\n') ;

tic
A1 = sparse (i, j, x, m, n) ;
t = toc ;
fprintf ('%12.4f sec : A = sparse (i, j, x, m, n) ;\n', t) ;

tic
A3 = gtb_build (ghb, i, j, x, m, n) ;
t = toc ;
fprintf ('%12.4f sec : A = %s.build (...), same inputs as built-in\n', ...
    t, gtb_name) ;
assert (gbtest_eq (A1, A3)) ;

tic
A4 = gtb_build (ghb, i, j, x, m, n, d) ;
t = toc ;
fprintf ('%12.4f sec : A = %s.build (..), same inputs/outputs as built-in\n',...
    t, gtb_name) ;
assert (gbtest_eq (A1, A4)) ;

tic
A2 = gtb_build (ghb, i0, j0, x, m, n, desc0) ;
t = toc ;
fprintf ('%12.4f sec : A = %s.build (i0,j0, ...), with i0 and j0 uint64\n',...
    t, gtb_name) ;
assert (gbtest_eq (A1, A2)) ;

i32 = uint32 (i0) ;
j32 = uint32 (j0) ;
tic
A5 = gtb_build (ghb, i32, j32, x, m, n, desc0) ;
t = toc ;
fprintf ('%12.4f sec : A = %s.build (i0,j0, ...), with i0 and j0 uint32\n',...
    t, gtb_name) ;
assert (gbtest_eq (A1, A5)) ;
clear A5

fprintf ('\nwith [I J] reversed/jumbled so that a sort is required:\n') ;
i = i (end:-1:1) ;
j = j (end:-1:1) ;
i (1:10) = i (randperm (10)) ;

i0 = uint64 (i) - 1 ;
j0 = uint64 (j) - 1 ;

tic
A1 = sparse (i, j, x, m, n) ;
t = toc ;
fprintf ('%12.4f sec : A = sparse (i, j, x, m, n) ;\n', t) ;

tic
A3 = gtb_build (ghb, i, j, x, m, n) ;
t = toc ;
fprintf ('%12.4f sec : A = %s.build (...), same inputs as built-in\n',...
    t, gtb_name) ;
assert (gbtest_eq (A1, A3)) ;

tic
A4 = gtb_build (ghb, i, j, x, m, n, d) ;
t = toc ;
fprintf ('%12.4f sec : A = %s.build (..), same inputs/outputs as built-in\n',...
    t, gtb_name) ;
assert (gbtest_eq (A1, A4)) ;

tic
A2 = gtb_build (ghb, i0, j0, x, m, n, desc0) ;
t = toc ;
fprintf ('%12.4f sec : A = %s.build (i0,j0, ...), with i0 and j0 uint64\n',...
    t, gtb_name) ;
assert (gbtest_eq (A1, A2)) ;

i32 = uint32 (i0) ;
j32 = uint32 (j0) ;
tic
A5 = gtb_build (ghb, i32, j32, x, m, n, desc0) ;
t = toc ;
fprintf ('%12.4f sec : A = %s.build (i0,j0, ...), with i0 and j0 uint32\n',...
    t, gtb_name) ;
assert (gbtest_eq (A1, A5)) ;
clear A5

fprintf ('\ngbtest28 (%d): all tests passed\n', ghb) ;

