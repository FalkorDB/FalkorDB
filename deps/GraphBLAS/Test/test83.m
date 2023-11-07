function test83
%TEST83 test GrB_assign with J=lo:0:hi, an empty list, and C_replace true

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% exercises the C_replace_phase of GB_assign.c

rng ('default') ;
n = 10 ;
A = sparse (ones (n,0)) ;
M = sparse (ones (n)) ;
M (1,1) = 0 ;
S = sparse (rand (n)) ;

J.begin = 1 ;
J.inc = 0 ;
J.end = 2 ;

I = [ ] ;

J1 = 2:0:3 ;

d.outp = 'replace' ;

C2 = GB_mex_assign (S, M, 'plus', A, I, J, d) ;

C1 = S ;
C1 (1,1) = 0 ;
assert (isequal (C1, C2.matrix)) ;

fprintf ('\ntest83: all tests passed\n') ;


