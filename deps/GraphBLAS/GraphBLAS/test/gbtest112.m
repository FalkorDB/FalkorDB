function gbtest112 (ghb)
%GBTEST112 test load and save

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = magic (5) ;
G = gtb (ghb, A) ;

if (ghb)
    filename = GhB.save (G) ;
else
    filename = GrB.save (G) ;
end

assert (isequal (filename, 'G.mat')) ;
H = gtb_load (ghb, 'G.mat') ;
assert (isequal (H, A)) ;
delete G.mat

if (ghb)
    filename = GhB.save (G+1) ;
else
    filename = GrB.save (G+1) ;
end

assert (isequal (filename, 'GrB_Matrix.mat')) ;
H = gtb_load (ghb, 'GrB_Matrix.mat') ;
assert (isequal (H, A+1)) ;
delete GrB_Matrix.mat

if (ghb)
    filename = GhB.save (A+1) ;
else
    filename = GrB.save (A+1) ;
end

assert (isequal (filename, 'GrB_Matrix.mat')) ;
H = gtb_load (ghb, 'GrB_Matrix.mat') ;
assert (isequal (H, A+1)) ;

K = gtb_load (ghb) ;
assert (isequal (H, K)) ;
delete GrB_Matrix.mat

f1 = [tempdir 'gbtest112_save.mat'] ;
save (f1, 'K') ;
K2 = gtb_load (ghb, f1) ;
assert (isequal (K, K2.K)) ;

fprintf ('\ngbtest112 (%d): all tests passed\n', ghb) ;

