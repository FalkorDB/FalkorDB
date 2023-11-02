function test212
%TEST212 test iso mask all zero

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

GB_mex_burble (1) ;

n = 10 ;
Cin.matrix = sprand (n, n, 0.5) ;
Cin.class = 'double' ;

A.matrix = sprand (n, n, 0.5) ;
A.class = 'double' ;

M.matrix = false (n,n) ;
M.pattern = logical (sprand (n, n, 0.5)) ;
M.iso = true ;

op.opname = 'sqrt' ;
op.optype = 'double' ;

desc = struct ('mask', 'complement') ;

C1 = GB_mex_apply  (Cin, M, [ ], op, A, [ ]) ;
C2 = GB_spec_apply (Cin, M, [ ], op, A, [ ]) ;
GB_spec_compare (C1, C2) ;

C1 = GB_mex_apply  (Cin, M, [ ], op, A, desc) ;
C2 = GB_spec_apply (Cin, M, [ ], op, A, desc) ;
GB_spec_compare (C1, C2) ;

M.matrix = true (n,n) ;

C1 = GB_mex_apply  (Cin, M, [ ], op, A, [ ]) ;
C2 = GB_spec_apply (Cin, M, [ ], op, A, [ ]) ;
GB_spec_compare (C1, C2) ;

C1 = GB_mex_apply  (Cin, M, [ ], op, A, desc) ;
C2 = GB_spec_apply (Cin, M, [ ], op, A, desc) ;
GB_spec_compare (C1, C2) ;

GB_mex_burble (0) ;
fprintf ('\ntest212: all tests passed\n') ;

