function gbtest112
%GBTEST112 test load and save

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

A = magic (5) ;
G = GrB (A) ;
filename = GrB.save (G) ;
assert (isequal (filename, 'G.mat')) ;
H = GrB.load ('G.mat') ;
assert (isequal (H, A)) ;
delete G.mat

filename = GrB.save (G+1) ;
assert (isequal (filename, 'GrB_Matrix.mat')) ;
H = GrB.load ('GrB_Matrix.mat') ;
assert (isequal (H, A+1)) ;
delete GrB_Matrix.mat

filename = GrB.save (A+1) ;
assert (isequal (filename, 'GrB_Matrix.mat')) ;
H = GrB.load ('GrB_Matrix.mat') ;
assert (isequal (H, A+1)) ;

K = GrB.load ;
assert (isequal (H, K)) ;
delete GrB_Matrix.mat

fprintf ('\ngbtest112: all tests passed\n') ;

