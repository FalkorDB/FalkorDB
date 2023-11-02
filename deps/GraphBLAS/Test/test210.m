function test210
%TEST210 test iso assign25: C<M,struct>=A, C empty, A dense, M structural

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 10 ;
A.matrix = sparse (ones (n, n)) ;
A.class = 'double' ;
A.iso = true ;

Cin = sparse (n,n) ;

M = spones (sprand (n, n, 0.5)) ;
desc = struct ('mask', 'structural') ;

C1 = GB_mex_assign (Cin, M, [ ], A, [ ], [ ], desc) ;
C2 = A.matrix .* M ;
assert (isequal (C1.matrix, C2)) ;

fprintf ('\ntest210: all tests passed\n') ;

