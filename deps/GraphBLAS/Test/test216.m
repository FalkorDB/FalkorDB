function test216
%TEST216 test C<A>=A, iso case

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

% GB_mex_burble (1) ;

n = 100 ;
Cin.matrix = pi * spones (sprand (n, n, 0.5)) ;
Cin.class = 'double' ;
Cin.iso = true ;
Cin.sparsity = 4 ;   % C is bitmap

A.matrix = pi * spones (sprand (n, n, 0.5)) ;
A.class  = 'double' ;
A.iso = true ;

desc = struct ('mask', 'structural') ;

C1 = GB_mex_assign_alias_mask (Cin, A, [ ]) ;
C2 = GB_spec_assign (Cin, A, [ ], A, [ ], [ ], [ ], false) ;
GB_spec_compare (C1, C2) ;

C1 = GB_mex_assign_alias_mask (Cin, A, desc) ;
C2 = GB_spec_assign (Cin, A, [ ], A, [ ], [ ], desc, false) ;
GB_spec_compare (C1, C2) ;

GB_mex_burble (0) ;
fprintf ('\ntest216: all tests passed\n') ;

