function gbtest90 (ghb, ghb2)
%GBTEST90 test [GrB,GhB].reduce
%
% c = GrB.reduce (monoid, A)
% c = GrB.reduce (monoid, A, desc)
% c = GrB.reduce (c, accum, monoid, A)
% c = GrB.reduce (c, accum, monoid, A, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
if (nargin < 2)
    ghb2 = ghb ;
end
gtb_name = gtb_prep (ghb) ;

C      = gtb (ghb2, pi) ;
accum  = '*' ;
monoid = '+' ;
A      = gtb_random (ghb2, 9, 9, 0.5) ;
desc   = struct ;

c = double (C) ;
a = double (A) ;

%----------------------------------------------------------------------
% c = GrB.reduce (monoid, A)
%----------------------------------------------------------------------

% 1 matrix: A
% 1 string: monoid

C2 = sum (A, 'all') ;
% works in R2019b; fails in R2018a:
% c2 = sum (a, 'all') ;
% works in R2018a:
c2 = sum (a (:)) ;
assert (isequal (c2, C2)) ;

C1 = gtb_reduce (ghb, monoid, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, A, monoid) ; assert (isequal (C1, C2)) ;

C1 = gtb_reduce (ghb, monoid, a) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, a, monoid) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% c = GrB.reduce (monoid, A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 1 string: monoid

C2 = sum (A, 'all') ;
c2 = sum (a (:)) ;
assert (isequal (c2, C2)) ;

C1 = gtb_reduce (ghb, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, A, monoid, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_reduce (ghb, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, a, monoid, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% c = GrB.reduce (c, accum, monoid, A)
%----------------------------------------------------------------------

% 2 matrices: c, A
% 2 strings: accum, monoid

C2 = C * sum (A, 'all') ;
c2 = c * sum (a (:)) ;
assert (isequal (c2, C2)) ;

C1 = gtb_reduce (ghb, C, accum, monoid, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, C, accum, A, monoid) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, C, A, accum, monoid) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, C, monoid, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, C, A, monoid) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, monoid, C, A) ; assert (isequal (C1, C2)) ;

C1 = gtb_reduce (ghb, c, accum, monoid, a) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, c, accum, a, monoid) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, c, a, accum, monoid) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, c, monoid, a) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, c, a, monoid) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, monoid, c, a) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% c = GrB.reduce (c, accum, monoid, A, desc)
%----------------------------------------------------------------------

% 2 matrices: c, A
% 2 strings: accum, monoid

C2 = C * sum (A, 'all') ;
c2 = c * sum (a (:)) ;
assert (isequal (c2, C2)) ;

C1 = gtb_reduce (ghb, C, accum, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, C, accum, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, C, A, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, C, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, C, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, monoid, C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_reduce (ghb, c, accum, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, c, accum, a, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, c, a, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, c, monoid, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, c, a, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_reduce (ghb, accum, monoid, c, a, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest90 (%d): all tests passed\n', ghb) ;

