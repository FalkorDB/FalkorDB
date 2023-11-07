function test225
%TEST225 test mask operations (GB_masker)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% GB_mex_burble (1) ;
rng ('default') ;
n = 1000 ;
S = sprand (n, n, 0.01) ;
Z = sprand (n, n, 0.5) ;
M = sprand (n, n, 0.01) ;
M (:,1) = sprand (n, 1, 0.9) ;
Z (:,1) = sprand (n, 1, 0.001) ;

d2.inp0 = 'tran' ;
d2.mask = 'complement' ;

C1 = GB_mex_transpose  (S, Z, [ ], M, d2) ;
C2 = GB_spec_transpose (S, Z, [ ], M, d2) ;
GB_spec_compare (C1, C2) ;

GB_mex_burble (0) ;
fprintf ('test225: all tests passed\n') ;

