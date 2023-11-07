function gbtest106
%GBTEST106 test build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

A = GrB.build (1:5, 1:5, true, 5, 5, 'xor') ;
assert (isequal (A, logical (speye (5)))) ;

fprintf ('\ngbtest106: all tests passed\n') ;

