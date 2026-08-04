function gbtest68 (ghb)
%GBTEST68 test isequal

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

s = gtb (ghb, pi) ;

assert (~isequal (s, magic (2))) ;
assert (~isequal (s, [pi pi])) ;
assert (~isequal (s, sparse (0))) ;

A = gtb (ghb, 2,2) ;
B = gtb (ghb, 2,2) ;
A (1,1) = 1 ;
B (2,2) = 1 ;
assert (~isequal (A, B)) ;

assert (~isequal (gtb (ghb, A, 'int8'), gtb (ghb, B, 'uint8'))) ;

fprintf ('gbtest68 (%d): all tests passed\n', ghb) ;

