function gbtest94 (ghb)
%GBTEST94 test [GrB,GhB].vreduce
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

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

C     = gtb_random (ghb, 9, 1, 0.5, 'range', [-1 1]) ;
M     = gtb_random (ghb, 9, 1, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = gtb_random (ghb, 9, 9, 0.5, 'range', [-1 1]) ;
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

C1 = gtb_vreduce (ghb, monoid, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, A, monoid) ; assert (isequal (C1, C2)) ;

C1 = gtb_vreduce (ghb, monoid, a) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, a, monoid) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.vreduce (monoid, A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 1 string: monoid

C2 = sum (A,2) ;
c2 = sum (a,2) ;
assert (isequal (c2, C2)) ;

C1 = gtb_vreduce (ghb, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, A, monoid, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_vreduce (ghb, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, a, monoid, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.vreduce (C, accum, monoid, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 strings: accum, monoid

C2 = C + sum (A,2) ;
c2 = c + sum (a,2) ;
assert (isequal (c2, C2)) ;

C1 = gtb_vreduce (ghb, C, accum, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, C, accum, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, C, A, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, accum, monoid, C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_vreduce (ghb, c, accum, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, c, accum, a, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, c, a, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, accum, monoid, c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.vreduce (C, M, monoid, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string: monoid

% C<M> = monoid (A)
T = sum (A,2) ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = sum (a,2) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_vreduce (ghb, C, M, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, C, monoid, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, monoid, C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, C, M, A, monoid, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_vreduce (ghb, c, m, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, c, monoid, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, monoid, c, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, c, m, a, monoid, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.vreduce (C, M, accum, monoid, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 strings: accum, monoid

% C<M> += monoid (A)

T = C + sum (A,2) ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = c + sum (a,2) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_vreduce (ghb, C, M, accum, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, C, accum, monoid, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, C, accum, M, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, C, accum, M, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, C, M, A, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, C, M, accum, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, accum, monoid, C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, accum, C, monoid, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, accum, C, M, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, accum, C, M, A, monoid, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_vreduce (ghb, c, m, accum, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, c, accum, monoid, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, c, accum, m, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, c, accum, m, a, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, c, m, a, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, c, m, accum, a, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, accum, monoid, c, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, accum, c, monoid, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, accum, c, m, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_vreduce (ghb, accum, c, m, a, monoid, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest94 (%d): all tests passed\n', ghb) ;

