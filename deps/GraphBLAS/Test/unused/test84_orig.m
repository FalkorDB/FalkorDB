function test84
%TEST84 test GrB_assign (row and column with C in CSR format)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
m = 10 ;
n = 20 ;

% create a CSR matrix
C0 = GB_spec_random (m, n, 0.5, 100, 'double', false, false) ;

Mcol = sparse (ones (m,1)) ; % spones (sprandn (m, 1, 0.5)) ;
Mrow = sparse (ones (n,1)) ; % spones (sprandn (n, 1, 0.5)) ;

Acol = sprandn (4, 1, 0.5)  ;
Arow = sprandn (4, 1, 0.5)  ;

J = [3 4 5 6] ;
J0 = uint64 (J) - 1 ;
I = 2 ;
I0 = uint64 (I) - 1 ;

% row assign
C1 = GB_mex_assign      (C0, Mrow, 'plus', Arow, I0, J0, [ ], 2) ;
C2 = GB_spec_Row_assign (C0, Mrow, 'plus', Arow, I,  J,  [ ]) ;
GB_spec_compare (C1, C2) ;

% col assign
C1 = GB_mex_assign      (C0, Mcol, 'plus', Acol, J0, I0, [ ], 1) ;
C2 = GB_spec_Col_assign (C0, Mcol, 'plus', Acol, J,  I,  [ ]) ;
GB_spec_compare (C1, C2) ;

fprintf ('\ntest84: all tests passed\n') ;


