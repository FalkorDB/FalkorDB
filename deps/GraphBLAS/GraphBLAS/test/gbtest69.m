function gbtest69 (ghb)
%GBTEST69 test flip

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = rand (10,8) ;
G = gtb (ghb, A) ;
assert (isequal (flip (A), flip (G))) ;
assert (isequal (flip (A,1), flip (G,1))) ;
assert (isequal (flip (A,2), flip (G,2))) ;
assert (isequal (flip (A,3), flip (G,3))) ;

assert (isequal (flip (A, gtb (ghb, 1)), flip (G,1))) ;
assert (isequal (flip (A, gtb (ghb, 2)), flip (G,2))) ;
assert (isequal (flip (A, gtb (ghb, 3)), flip (G,3))) ;

A = rand (10,1) ;
G = gtb (ghb, A) ;
assert (isequal (flip (A), flip (G))) ;
assert (isequal (flip (A,1), flip (G,1))) ;
assert (isequal (flip (A,2), flip (G,2))) ;
assert (isequal (flip (A,3), flip (G,3))) ;

assert (isequal (flip (A, gtb (ghb,1)), flip (G,1))) ;
assert (isequal (flip (A, gtb (ghb,2)), flip (G,2))) ;
assert (isequal (flip (A, gtb (ghb,3)), flip (G,3))) ;

A = rand (1,9) ;
G = gtb (ghb, A) ;
assert (isequal (flip (A), flip (G))) ;
assert (isequal (flip (A,1), flip (G,1))) ;
assert (isequal (flip (A,2), flip (G,2))) ;
assert (isequal (flip (A,3), flip (G,3))) ;

assert (isequal (flip (A, gtb (ghb,1)), flip (G,1))) ;
assert (isequal (flip (A, gtb (ghb,2)), flip (G,2))) ;
assert (isequal (flip (A, gtb (ghb,3)), flip (G,3))) ;

fprintf ('gbtest69 (%d): all tests passed\n', ghb) ;

