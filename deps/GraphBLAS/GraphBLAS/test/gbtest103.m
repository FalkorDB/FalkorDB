function gbtest103
%GBTEST103 test iso matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
n = 2^52 ;
A = GrB.ones (n,n)  %#ok<NOPRT>
assert (A (n/2, n) == 1) ;

fprintf ('\ngbtest103: all tests passed\n') ;

