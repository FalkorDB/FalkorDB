function gbtest69
%GBTEST69 test flip

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

A = rand (10,8) ;
G = GrB (A) ;
assert (isequal (flip (A), flip (G))) ;
assert (isequal (flip (A,1), flip (G,1))) ;
assert (isequal (flip (A,2), flip (G,2))) ;
assert (isequal (flip (A,3), flip (G,3))) ;

assert (isequal (flip (A,GrB (1)), flip (G,1))) ;
assert (isequal (flip (A,GrB (2)), flip (G,2))) ;
assert (isequal (flip (A,GrB (3)), flip (G,3))) ;

A = rand (10,1) ;
G = GrB (A) ;
assert (isequal (flip (A), flip (G))) ;
assert (isequal (flip (A,1), flip (G,1))) ;
assert (isequal (flip (A,2), flip (G,2))) ;
assert (isequal (flip (A,3), flip (G,3))) ;

assert (isequal (flip (A, GrB(1)), flip (G,1))) ;
assert (isequal (flip (A, GrB(2)), flip (G,2))) ;
assert (isequal (flip (A, GrB(3)), flip (G,3))) ;

A = rand (1,9) ;
G = GrB (A) ;
assert (isequal (flip (A), flip (G))) ;
assert (isequal (flip (A,1), flip (G,1))) ;
assert (isequal (flip (A,2), flip (G,2))) ;
assert (isequal (flip (A,3), flip (G,3))) ;

assert (isequal (flip (A,GrB(1)), flip (G,1))) ;
assert (isequal (flip (A,GrB(2)), flip (G,2))) ;
assert (isequal (flip (A,GrB(3)), flip (G,3))) ;

fprintf ('gbtest69: all tests passed\n') ;

