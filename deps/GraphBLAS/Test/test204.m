function test204
%TEST204 test iso diag

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% GB_mex_burble (1) ;
rng ('default') ;
n = 10 ;
A.matrix = pi * spones (sprandn (n, n, 0.5)) ;
A.class = 'double' ;
A.iso = true ;

v1 = GB_mex_vdiag (A) ;
v2 = diag (A.matrix) ;
assert (isequal (v1.matrix, v2)) ;

C1 = GB_mex_mdiag (v1) ;
C2 = diag (v2) ;
assert (isequal (C1.matrix, C2)) ;

GB_mex_burble (0) ;
fprintf ('test204: all tests passed\n') ;

