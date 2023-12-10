function test217
%TEST217 test C<repl>(I,J)=A, bitmap assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

GB_mex_burble (1) ;

n = 100 ;
Cin.matrix = sprand (n, n, 0.5) ;
Cin.class = 'double' ;
Cin.sparsity = 4 ;   % C is bitmap

m = 10 ;
I1 = randperm (n, 10) ;
J1 = randperm (n, 10) ;
I0 = uint64 (I1) - 1 ;
J0 = uint64 (J1) - 1 ;

A.matrix = sprand (m, m, 0.5) ;
A.class  = 'double' ;
A.sparsity = 4 ;   % A is bitmap

desc = struct ('outp', 'replace') ;
C1 = GB_mex_assign  (Cin, [ ], [ ], A, I0, J0, desc)
C2 = GB_spec_assign (Cin, [ ], [ ], A, I1, J1, desc, false) ;
GB_spec_compare (C1, C2) ;

desc = struct ('outp', 'replace', 'mask', 'complement') ;
C1 = GB_mex_assign  (Cin, [ ], [ ], A, I0, J0, desc)
C2 = GB_spec_assign (Cin, [ ], [ ], A, I1, J1, desc, false) ;
GB_spec_compare (C1, C2) ;

GB_mex_burble (0) ;
fprintf ('\ntest217: all tests passed\n') ;

