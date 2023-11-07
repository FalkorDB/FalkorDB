function test174
%TEST174 bitmap assignment, C<!,repl>+=A

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 100 ;
A = GB_spec_random (n, n, 0.05, 1, 'double') ;
A.sparsity = 2 ;

C = GB_spec_random (n, n, 0.05, 1, 'double') ;
C.sparsity = 4 ;

desc.mask = 'complement' ;
desc.outp = 'replace' ;

% C<!,repl> += A
C1 = GB_spec_assign (C, [ ], 'plus', A, [ ], [ ], desc, false) ;
C2 = GB_mex_assign  (C, [ ], 'plus', A, [ ], [ ], desc, false) ;
GB_spec_compare (C1, C2) ;

k = 10 ;
I = randperm (n, k) ;
J = randperm (n, k) ;
I0 = uint64 (I) - 1 ;
J0 = uint64 (J) - 1 ;

A = GB_spec_random (k, k, 0.05, 1, 'double') ;
A.sparsity = 2 ;

% C<!,repl>(I,J) += A
C1 = GB_spec_assign (C, [ ], 'plus', A, I,  J,  desc, false) ;
C2 = GB_mex_assign  (C, [ ], 'plus', A, I0, J0, desc, false) ;
GB_spec_compare (C1, C2) ;

I = 2 ;
I0 = uint64 (I) - 1 ;
Arow = sprand (n, 1, 0.5) ;
Acol = sprand (n, 1, 0.5) ;

% C<!,repl>(i,:) = A
C1 = GB_spec_Row_assign (C, [ ], 'plus', Arow, I,  [ ],  desc) ;
C2 = GB_mex_assign      (C, [ ], 'plus', Arow, I0, [ ], desc, 2) ;
GB_spec_compare (C1, C2) ;

% C<!,repl>(:,i) = A
C1 = GB_spec_Col_assign (C, [ ], 'plus', Acol, [ ], I,  desc) ;
C2 = GB_mex_assign      (C, [ ], 'plus', Acol, [ ], I0, desc, 1) ;
GB_spec_compare (C1, C2) ;

fprintf ('test174: all tests passed\n') ;
