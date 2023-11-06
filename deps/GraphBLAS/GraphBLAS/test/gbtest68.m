function gbtest68
%GBTEST68 test isequal

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

s = GrB (pi) ;

assert (~isequal (s, magic (2))) ;
assert (~isequal (s, [pi pi])) ;
assert (~isequal (s, sparse (0))) ;

A = GrB (2,2) ;
B = GrB (2,2) ;
A (1,1) = 1 ;
B (2,2) = 1 ;
assert (~isequal (A, B)) ;

assert (~isequal (GrB (A, 'int8'), GrB (B, 'uint8'))) ;

fprintf ('gbtest68: all tests passed\n') ;

