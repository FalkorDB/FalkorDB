function test205
%TEST205 test iso kron

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% GB_mex_burble (1) ;
rng ('default') ;
n = 10 ;
A.matrix = pi * spones (sprandn (n, n, 0.5)) ;
A.class = 'double' ;
A.iso = true ;

B.matrix = 3 * spones (sprandn (n, n, 0.5)) ;
B.class = 'double' ;
B.iso = true ;

op_times.opname = 'times' ;
op_times.optype = 'double' ;

Cin = sparse (n*n,n*n) ;
C1 = GB_mex_kron (Cin, [ ], [ ], op_times, A, B, [ ]) ;
C2 = kron (A.matrix, B.matrix) ;
err = norm (C1.matrix - C2, 1) ;
assert (err < 1e-12) 
assert (C1.iso) ;

A.matrix = pi * ones (n,n) ;
B.matrix = 3 * ones (n,n) ;

Cin = sparse (n*n,n*n) ;
C1 = GB_mex_kron (Cin, [ ], [ ], op_times, A, B, [ ]) ;
C2 = kron (A.matrix, B.matrix) ;
err = norm (C1.matrix - C2, 1) ;
assert (err < 1e-12) 
assert (C1.iso) ;

GB_mex_burble (0) ;
fprintf ('test205: all tests passed\n') ;

