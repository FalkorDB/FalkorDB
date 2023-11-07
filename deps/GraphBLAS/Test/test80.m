function test80
%TEST80 rerun test06 with different matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
n = 33 ;
A = speye (n) ;
A (:,1) = rand (n,1) ;
B = speye (n) ;
B (1,:) = rand (n,1) ;

test06 (A, B, 0) ;

A (:,2:3) = 0 ;
A (1,2) = 1 ;
B (2:3,:) = 0 ;
B (2,1) = 1 ;

test06 (A, B, 0) ;

fprintf ('\ntest80: all test passed\n') ;

