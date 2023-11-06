function test133
%TEST133 test mask operations (GB_masker)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest133: test the mask operation\n') ;

n = 1000 ;
S = sprand (n, n, 0.01) ;
Z = sprand (n, n, 0.5) ;
M = sprand (n, n, 0.01) ;

dtn.inp0 = 'tran' ;

C1 = GB_mex_transpose  (S, M, [ ], Z, dtn) ;
C2 = GB_spec_transpose (S, M, [ ], Z, dtn) ;
GB_spec_compare (C1, C2) ;

d2.inp0 = 'tran' ;
d2.mask = 'complement' ;

C1 = GB_mex_transpose  (S, Z, [ ], M, d2) ;
C2 = GB_spec_transpose (S, Z, [ ], M, d2) ;
GB_spec_compare (C1, C2) ;

D = sprand (n, n, 0.8) ;
D (:, 2:end) = 0 ;

n2 = floor (n/2) ;
S (n2:end, :) = 0 ;
M = sparse (n, n) ;
M (n,1) = 1 ;

C1 = GB_mex_transpose  (D, M, [ ], S, d2) ;
C2 = GB_spec_transpose (D, M, [ ], S, d2) ;
GB_spec_compare (C1, C2) ;

fprintf ('\ntest133: all tests passed\n') ;


