function test132
%TEST132 test GrB_*_setElement and GrB_*_*build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% simplified from test45

fprintf ('\ntest132 ------------------ testing GrB_setElement and _build\n') ;

% debug_on ;

rng ('default') ;
A = sparse (rand (3,2)) ;

C = GB_mex_setElement (A, uint64(0), uint64(0), 42.1, true) ;

A = rand (3,2) ;
A (2,2) = 0 ;
A = sparse (A)  ;

C = GB_mex_setElement (A, uint64(1), uint64(1), 99, true) ;
GB_spok (C.matrix) ;

A = sprand (67, 67, 0.1) ;

[m n] = size (A) ;

ntuples = 1000 ;
C1 = A ;
I = 1 + floor (m * rand (ntuples, 1)) ;
J = 1 + floor (n * rand (ntuples, 1)) ;
X = 100 * rand (ntuples, 1) ;
I0 = uint64 (I)-1 ;
J0 = uint64 (J)-1 ;

for k = 1:ntuples
    C1 (I (k), J (k)) =  X (k) ;
end

A2 = A ;
C3 = GB_mex_setElement (A2, I0, J0, X, true) ;
C4 = GB_mex_setElement (A2, I0, J0, X, true, true) ;
assert (GB_spok (C3.matrix) == 1)
assert (GB_spok (C4.matrix) == 1)

assert (isequal (C3.matrix, C1)) ;
assert (isequal (C4.matrix, C1)) ;

ntuples = 10 ;
I = (m * ones (ntuples, 1)) ;
J = (n * ones (ntuples, 1)) ;
I0 = uint64 (I)-1 ;
J0 = uint64 (J)-1 ;
X = pi * ones (ntuples, 1) ;

A (m,n) = 0 ;
A (:, 5:67) = 0 ;

C1 = A ;
for k = 1:ntuples
    C1 (I (k), J (k)) =  X (k) ;
end

clear A2
A2.matrix = A ;
A2.is_hyper = true ;
C3 = GB_mex_setElement (A2, I0, J0, X, true) ;
C4 = GB_mex_setElement (A2, I0, J0, X, true) ;
assert (GB_spok (C3.matrix) == 1)
assert (isequal (C3.matrix, C1)) ;
assert (isequal (C4.matrix, C1)) ;

fprintf ('\ntest132: all tests passed\n') ;

