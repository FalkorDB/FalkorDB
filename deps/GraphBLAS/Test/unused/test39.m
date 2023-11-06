function test39(use_ssget)
%TEST39 performance test for GrB_transpose

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest39 performance tests : GrB_transpose \n') ;

if (nargin < 1)
    use_ssget = true ;
end

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

rng ('default') ;

if (use_ssget)
    try
        Prob = ssget (939)
        A = Prob.A ;
    catch
        use_ssget = false ;
    end
end

if (~use_ssget)
    fprintf ('not using ssget\n') ;
    n = 72000 ;
    nz = 29e6 ;
    A = sprandn (n, n, nz/n^2) ;
end

[m n] = size (A) ;
Cin = sprandn (n, m, 0.000001) ;
A (1,2) =1 ;
Empty = sparse (n, m) ;

fprintf ('\n===============================================================n') ;
fprintf ('\nC = A''\n') ;
tic
C1 = A' ;
toc
tm = toc ;

fprintf ('GraphBLAS, transpose :\n') ;
tic
C = GB_mex_transpose (Empty, [ ], [ ], A) ;
tg = toc ;
fprintf ('GraphBLAS time: %g\n', tg) ;
assert (isequal (C1, C.matrix)) ;
fprintf ('speedup over built-in: %g\n\n', tm/tg) ;

fprintf ('\n===============================================================n') ;
fprintf ('\nGraphBLAS: C = (single) A'' compared with C=A'' in built-in\n') ;
clear Empty_struct
Empty_struct.matrix = sparse (n, m) ;
Empty_struct.class = 'single' ;

tic
C1 = A' ;
toc
tm = toc ;

fprintf ('GraphBLAS, transpose:\n') ;
% C = A'
tic
C2 = GB_mex_transpose (Empty_struct, [ ], '', A) ;
tg = toc ;
fprintf ('GraphBLAS time: %g\n', tg) ;
fprintf ('speedup over built-in: %g\n\n', tm/tg) ;

[I1, J1, X1] = find (C1) ;
[I2, J2, X2] = find (C2.matrix) ;
clear C2

assert (isequal (I1, I2)) ;
assert (isequal (J1, J2)) ;
assert (isequal (single(X1), X2)) ;

fprintf ('\n===============================================================n') ;
fprintf ('\nC = Cin + A''\n') ;
tic
C1 = Cin + A' ;
toc
tm = toc ;

fprintf ('GraphBLAS, transpose and then accum with GB_add:\n') ;
% C = Cin + A'
tic
C = GB_mex_transpose (Cin, [ ], 'plus', A) ;
tg = toc ;
fprintf ('GraphBLAS time: %g\n', tg) ;
assert (isequal (C1, C.matrix)) ;
fprintf ('speedup over built-in: %g\n\n', tm/tg) ;

fprintf ('\n===============================================================n') ;
fprintf ('\nC = A + B\n') ;

B = sprandn (m, n, 0.00001) ;
fprintf ('nnz (A) = %d nnz (B) = %d\n', nnz (A), nnz (B)) ;

tic
C1 = A + B ;
toc
tm = toc ;

D = struct ('inp0', 'tran') ;

fprintf ('\nusing accum and subassign, then GB_wait:\n') ;
tic
C2 = GB_mex_transpose (A, [ ], 'plus', B, D) ;
tg = toc ;
fprintf ('GraphBLAS time: %g\n', tg) ;
fprintf ('speedup over built-in: %g\n\n', tm/tg) ;
assert (isequal (C1, C2.matrix)) ;

fprintf ('\nusing accum and GB_add:\n') ;
tic
C2 = GB_mex_transpose (B, [ ], 'plus', A, D) ;
tg = toc ;
fprintf ('GraphBLAS time: %g\n', tg) ;
fprintf ('speedup over built-in: %g\n\n', tm/tg) ;
assert (isequal (C1, C2.matrix)) ;

fprintf ('\nvia GB_add and then accum:\n') ;
clear Cin
Cin = sparse (m,n) ;
tic
C3 = GB_mex_Matrix_eWiseAdd (Cin, [ ], '', 'plus', A, B) ;
tg = toc ;
fprintf ('GraphBLAS time: %g\n', tg) ;
fprintf ('speedup over built-in: %g\n\n', tm/tg) ;
assert (isequal (C1, C3.matrix)) ;

fprintf ('\nvia GB_add:\n') ;
tic
C4 = GB_mex_AplusB (A, B, 'plus') ;
tg = toc ;
fprintf ('GraphBLAS time: %g\n', tg) ;
fprintf ('speedup over built-in: %g\n\n', tm/tg) ;
assert (isequal (C1, C4)) ;

fprintf ('\nvia GB_add:\n') ;
tic
C4 = GB_mex_AplusB (B, A, 'plus') ;
tg = toc ;
fprintf ('GraphBLAS time: %g\n', tg) ;
fprintf ('speedup over built-in: %g\n\n', tm/tg) ;
assert (isequal (C1, C4)) ;

fprintf ('\n===============================================================n') ;
fprintf ('\nC = Cin + A + B\n') ;

Cin = sprandn (m, n, 0.0001) ;

tic
C1 = Cin + A + B ;
toc
tm1 = toc ;

tic
C1 = (Cin + A) + B ;
toc
tm2 = toc ;

tic
C1 = Cin + (A + B) ;
toc
tm3 = toc ;

tic
C1 = (Cin + B) + A ;
toc
tm5 = toc ;

tic
C3 = GB_mex_Matrix_eWiseAdd (Cin, [ ], 'plus', 'plus', A, B) ;
tg = toc ;
fprintf ('GraphBLAS time: %g\n', tg) ;
fprintf ('speedup over built-in: %g\n\n', tm1/tg) ;

assert (isequal (C1, C3.matrix)) ;

fprintf ('\nvia two GB_add: (Cin+A)+B:\n') ;
tic
C4 = GB_mex_AplusB (Cin, A, 'plus') ;
tg1 = toc ;
tic
C4 = GB_mex_AplusB (C4, B, 'plus') ;
tg2 = toc ;
fprintf ('GraphBLAS time: %g\n', tg1+tg2) ;
fprintf ('speedup over built-in: %g\n\n', tm2/(tg1+tg2)) ;
assert (isequal (C1, C4)) ;;

fprintf ('\nvia two GB_add: (Cin+(A+B)):\n') ;
tic
C4 = GB_mex_AplusB (A, B, 'plus') ;
tg1 = toc ;
tic
C4 = GB_mex_AplusB (C4, Cin, 'plus') ;
tg2 = toc ;
fprintf ('GraphBLAS time: %g\n', tg1+tg2) ;
fprintf ('speedup over built-in: %g\n\n', tm3/(tg1+tg2)) ;
assert (isequal (C1, C4)) 

fprintf ('\nvia two GB_add: (Cin+B)+A)):\n') ;
tic ;
C4 = GB_mex_AplusB (Cin, B, 'plus') ;
tg1 = toc ;

tic
C4 = GB_mex_AplusB (C4, A, 'plus') ;
tg2 = toc ;

fprintf ('GraphBLAS time: %g\n', tg1+tg2) ;
fprintf ('speedup over built-in: %g\n\n', tm5/(tg1+tg2)) ;
assert (isequal (C1, C4)) ;;

nthreads_set (save, save_chunk) ;

fprintf ('\ntest39: all tests passed\n') ;

