function test46
%TEST46 performance test of GxB_subassign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n--------------performance test GB_mex_subassign\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;
debug_off

dt = struct ('inp0', 'tran') ;

rng ('default') ;

A = sparse (rand (3,4)) ;
I = uint64 (0:2) ;
J = uint64 (0:3) ;
C = A ;

C0 = sprandn (length(I), length (J), 0.5) ;
C1 = C ;
C1 (I+1,J+1) = C0 ;
CC = GB_mex_subassign (C, [],[], C0, I, J, []) ;
assert (isequal (C1, CC.matrix))

C0 = sprandn (length(I), length (J), 0.5)' ;
C1 = C ;
C1 (I+1,J+1) = C0' ;
CC = GB_mex_subassign (C, [],[], C0, I, J, dt) ;
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
            C2 = GB_mex_subassign (C, [], '', A, I0, J0, []) ;

            C3 = C ;
            C3 (I,J) = C3 (I,J) + A ;
            C4 = GB_mex_subassign (C, [], 'plus', A, I0, J0, []) ;

            assert (isequal (C3, C4.matrix))
        end
    end
end


Prob = ssget (2662) ;
A = Prob.A ;
% n = 2^21 ; A = A (1:n,1:n) ;
% A = A (:) ;
A = [A A ; A A] ;
A = [A A ; A A] ;
% A = [A A ; A A] ;
%A = [A A ; A A] ;
%A = [A A ; A A] ;
C = A ; 
C (1,1) = 1 ;
[m n] = size (A) ;
fprintf ('size: %d %d\n', m, n) ;

ni = min (size (A, 1), 5500) ;
nj = min (size (A, 2), 7000) ;
B = sprandn (ni, nj, 0.001) ;
% I = randperm (m) ; I = I (1:ni) ;
% J = randperm (n) ; J = J (1:nj) ;
I = randperm (m,ni) ;
J = randperm (n,nj) ;
fprintf ('nnzB: %g\n', nnz (B)) ;

fprintf ('\nC(I,J)=B, builtin start:\n')
tic
C (I,J) = B ;
toc

I0 = uint64 (I-1) ;
J0 = uint64 (J-1) ;

C2 = A;
C2 (1,1) =1 ;

fprintf ('GraphBLAS start:\n')

for nthreads = [1 20 40]
    nthreads_set (nthreads) ;
    tic
    C3 = GB_mex_subassign (C2, [], [], B, I0, J0, []) ;
    tg = toc ;
    fprintf ('%d threads, GB time: %g\n', nthreads, tg) ;
    assert (isequal (C, C3.matrix)) ;
end

% A = Prob.A ;
% A = A (:) ;
% A = [A A] ;
% n = 2^21 ; A = A (1:n,1:n) ;
C = A ;
C (1,1) = 1 ;

fprintf ('\nC(I,J)+=B, builtin start:\n')
tic
C (I,J) = C (I,J) + B ;
toc

C2 = A ;
C2 (1,1) = 1 ;

fprintf ('GraphBLAS start:\n')

for nthreads = [1 20 40]
    nthreads_set (nthreads) ;
    tic
    C3 = GB_mex_subassign (C2, [], 'plus', B, I0, J0, []) ;
    tg = toc ;
    fprintf ('%d threads, GB time: %g\n', nthreads, tg) ;
    assert (isequal (C, C3.matrix)) ;
end

nthreads_set (save, save_chunk) ;

fprintf ('\ntest46: all tests passed\n') ;

