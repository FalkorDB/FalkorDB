function gbtest114
%GBTEST114 test kron with iso matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

A = GrB.random (5, 10, 0.4) ;
B = GrB.ones (3, 2) ;

C1 = kron (A, B) ;
C2 = kron (double (A), double (B)) ;
assert (isequal (C1, C2)) ;

C1 = kron (B, A) ;
C2 = kron (double (B), double (A)) ;
assert (isequal (C1, C2)) ;

C1 = kron (B, B) ;
C2 = kron (double (B), double (B)) ;
assert (isequal (C1, C2)) ;

fprintf ('\ngbtest114: all tests passed\n') ;


