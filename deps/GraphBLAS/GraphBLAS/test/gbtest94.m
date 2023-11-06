function gbtest94
%GBTEST94 test GrB.vreduce
%
% C = GrB.vreduce (monoid, A)
% C = GrB.vreduce (monoid, A, b)
% C = GrB.vreduce (monoid, A, b, desc)
%
% C = GrB.vreduce (C, accum, monoid, A)
% C = GrB.vreduce (C, accum, monoid, A, b)
% C = GrB.vreduce (C, accum, monoid, A, b, desc)
%
% C = GrB.vreduce (C, M, monoid, A)
% C = GrB.vreduce (C, M, monoid, A, b)
% C = GrB.vreduce (C, M, monoid, A, b, desc)
%
% C = GrB.vreduce (C, M, accum, monoid, A)
% C = GrB.vreduce (C, M, accum, monoid, A, b)
% C = GrB.vreduce (C, M, accum, monoid, A, b, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (9, 1, 0.5, 'range', [-1 1]) ;
M     = GrB.random (9, 1, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GrB.random (9, 9, 0.5, 'range', [-1 1]) ;
desc  = struct ;

monoid = '+' ;

c = double (C) ;
m = logical (M) ;
a = double (A) ;

%----------------------------------------------------------------------
% C = GrB.vreduce (monoid, A)
%----------------------------------------------------------------------

% 1 matrix: A
% 1 string: monoid

C2 = sum (A,2) ;
c2 = sum (a,2) ;
assert (isequal (c2, C2)) ;

C1 = GrB.vreduce (monoid, A) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (A, monoid) ; assert (isequal (C1, C2)) ;

C1 = GrB.vreduce (monoid, a) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (a, monoid) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.vreduce (monoid, A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 1 string: monoid

C2 = sum (A,2) ;
c2 = sum (a,2) ;
assert (isequal (c2, C2)) ;

C1 = GrB.vreduce (monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (A, monoid, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.vreduce (monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (a, monoid, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.vreduce (C, accum, monoid, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 strings: accum, monoid

C2 = C + sum (A,2) ;
c2 = c + sum (a,2) ;
assert (isequal (c2, C2)) ;

C1 = GrB.vreduce (C, accum, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (C, accum, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (C, A, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (accum, monoid, C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.vreduce (c, accum, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (c, accum, a, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (c, a, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (accum, monoid, c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.vreduce (C, M, monoid, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string: monoid

% C<M> = monoid (A)
T = sum (A,2) ;
C2 = C ;
C2 (M) = T (M) ;

t = sum (a,2) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.vreduce (C, M, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (C, monoid, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (monoid, C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (C, M, A, monoid, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.vreduce (c, m, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (c, monoid, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (monoid, c, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (c, m, a, monoid, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.vreduce (C, M, accum, monoid, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 strings: accum, monoid

% C<M> += monoid (A)

T = C + sum (A,2) ;
C2 = C ;
C2 (M) = T (M) ;

t = c + sum (a,2) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.vreduce (C, M, accum, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (C, accum, monoid, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (C, accum, M, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (C, accum, M, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (C, M, A, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (C, M, accum, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (accum, monoid, C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (accum, C, monoid, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (accum, C, M, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (accum, C, M, A, monoid, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.vreduce (c, m, accum, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (c, accum, monoid, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (c, accum, m, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (c, accum, m, a, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (c, m, a, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (c, m, accum, a, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (accum, monoid, c, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (accum, c, monoid, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (accum, c, m, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.vreduce (accum, c, m, a, monoid, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest94: all tests passed\n') ;

