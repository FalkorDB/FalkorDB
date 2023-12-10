function test64
%TEST64 test GxB_*_subassign, scalar expansion, with and without duplicates

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n ------------------- quick test of GxB_*_subassign_scalar\n') ;

Corig = sparse (rand (5,4)) ;

% no accum, no duplicates in I
C = Corig ;
I = [2 3 5] ;
J = [1 3] ;
C (I,J) = 100 ;
I0 = uint64 (I-1) ;
J0 = uint64 (J-1) ;
C2 = GB_mex_subassign (Corig, [], [], sparse (100), I0, J0, []) ;
assert (isequal (C, C2.matrix)) 

% no accum, with duplicates in I
C = Corig ;
I = [2 2 5] ;
J = [1 3] ;
C (I,J) = 100 ;
I0 = uint64 (I-1) ;
J0 = uint64 (J-1) ;
C2 = GB_mex_subassign (Corig, [], [], sparse (100), I0, J0, []) ;
assert (isequal (C, C2.matrix)) 

% accum 'plus', no duplicates in I
C = Corig ;
I = [2 3 5] ;
J = [1 3] ;
C (I,J) = C (I,J) + 100 ;
I0 = uint64 (I-1) ;
J0 = uint64 (J-1) ;
C2 = GB_mex_subassign (Corig, [], 'plus', sparse (100), I0, J0, []) ;
assert (isequal (C, C2.matrix)) 

fprintf ('\ntest64: all tests passed\n') ;

