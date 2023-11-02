function gbtest73
%GBTEST73 test GrB.normdiff

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

x = rand (5, 1) ;
y = rand (5, 1) ;
e1 = GrB.normdiff (x, y) ;
e2 = norm (x-y) ;
assert (abs (e1 - e2) < 1e-12) ;

fprintf ('gbtest73: all tests passed\n') ;

