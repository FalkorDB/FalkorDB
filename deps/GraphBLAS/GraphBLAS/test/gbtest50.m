function gbtest50
%GBTEST50 test GrB.ktruss and GrB.tricount

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

% The MathWorks has mangled the Harwell-Boeing west0479 matrix,
% by reducing the precision of its entries, and dropping one entry.
% The correct version is in the HB/west0479 matrix at sparse.tamu.edu.

load west0479_correct.txt
west0479 = spconvert (west0479_correct) ;

A = GrB.offdiag (west0479) ;
A = A+A' ;
C3a  = GrB.ktruss (A) ;
C3  = GrB.ktruss (A, 3) ;
assert (isequal (C3a, C3)) ;
C3  = GrB.ktruss (A, 3, 'check') ;
assert (isequal (C3a, C3)) ;

ntriangles = sum (C3, 'all') / 6 ;
assert (ntriangles == 237) ;

C4a = GrB.ktruss (A, 4) ;
C4b = GrB.ktruss (C3, 4) ;          % this is faster
assert (isequal (C4a, C4b)) ;

nt2 = GrB.tricount (A) ;
assert (ntriangles == nt2) ;

d = GrB.entries (A, 'col', 'degree') ;
nt2 = GrB.tricount (A, d) ;
assert (ntriangles == nt2) ;

nt2 = GrB.tricount (A, 'check', d) ;
assert (ntriangles == nt2) ;

nt2 = GrB.tricount (A, d, 'check') ;
assert (ntriangles == nt2) ;

rng ('default') ;
for k = 1:200
    if (mod (k, 10) == 1)
        fprintf ('.') ;
    end
    n = 10000 ;
    G = GrB.eye (10000) ;
    j = randperm (n, 10) ;
    G (:,j) = 1 ;
    G (j,:) = 1 ;
    nt = GrB.tricount (G) ; %#ok<NASGU>
end

fprintf ('\ngbtest50: all tests passed\n') ;

