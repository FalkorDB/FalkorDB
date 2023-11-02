function test46b
%TEST46B performance test of GrB_assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n--------------performance test GB_mex_assign\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

dt = struct ('inp0', 'tran') ;

rng ('default') ;

A = sparse (rand (3,4)) ;
I = uint64 (0:2) ;
J = uint64 (0:3) ;
C = A ;

C0 = sprandn (length(I), length (J), 0.5) ;
C1 = C ;
C1 (I+1,J+1) = C0 ;
CC = GB_mex_assign (C, [],[], C0, I, J, []) ;
assert (isequal (C1, CC.matrix))

C0 = sprandn (length(I), length (J), 0.5)' ;
C1 = C ;
C1 (I+1,J+1) = C0' ;
CC = GB_mex_assign (C, [],[], C0, I, J, dt) ;
assert (isequal (C1, CC.matrix))

for trial = 1:100

    for m = [1 10 100]
        for n = [1 10 100]

            C = sprandn (m, n, 0.1) ;

            ni = double (irand (1, m)) ;
            nj = double (irand (1, n)) ;
            I = randperm (m) ;
            J = randperm (n) ;
            I = I (1:ni) ;
            J = J (1:nj) ;
            I0 = uint64 (I-1) ;
            J0 = uint64 (J-1) ;

            A = sprandn (ni, nj, 0.1) ;

            C1 = C ;
            C1 (I,J) = A ;
            C2 = GB_mex_assign (C, [], '', A, I0, J0, []) ;

            C3 = C ;
            C3 (I,J) = C3 (I,J) + A ;
            C4 = GB_mex_assign (C, [], 'plus', A, I0, J0, []) ;

            assert (isequal (C3, C4.matrix))
        end
    end
end


Prob = ssget (2662) ;
A = Prob.A ;
C = A ; 
C (1,1) = 1 ;
[m n] = size (A) ;

ni = 5500 ;
nj = 7000 ;
B = sprandn (ni, nj, 0.001) ;
I = randperm (m) ; I = I (1:ni) ;
J = randperm (n) ; J = J (1:nj) ;
fprintf ('nnzB: %g\n', nnz (B)) ;

fprintf ('built-in start:\n')
tic
C (I,J) = B ;
toc

I0 = uint64 (I-1) ;
J0 = uint64 (J-1) ;

C2 = A;
C2 (1,1) =1 ;

fprintf ('GraphBLAS start:\n')
tic
C3 = GB_mex_assign (C2, [], [], B, I0, J0, []) ;
toc

assert (isequal (C, C3.matrix)) ;

A = Prob.A ;
C = A ;
C (1,1) = 1 ;

fprintf ('built-in start:\n')
tic
C (I,J) = C (I,J) + B ;
toc

C2 = A ;
C2 (1,1) = 1 ;

fprintf ('GraphBLAS start:\n')
tic
C3 = GB_mex_assign (C2, [], 'plus', B, I0, J0, []) ;
toc

assert (isequal (C, C3.matrix)) ;

nthreads_set (save, save_chunk) ;

fprintf ('\ntest46b: all tests passed\n') ;
