function gbtest143
%GBTEST143 test GhB.reduce (inplace usage)
%
% GhB.reduce (c, op, A)                    c = op (A)
% GhB.reduce (c, accum, op, A)             c += op (A)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C      = GhB (pi) ;
accum  = '*' ;
monoid = '+' ;
A      = GhB.random (9, 9, 0.5) ;
desc   = struct ;
C0     = GhB (1, 1) ;

%----------------------------------------------------------------------
% GhB.reduce (c, monoid, A)
%----------------------------------------------------------------------

% 2 matrices: c, A
% 1 string: monoid

C2 = sum (A, 'all') ;
C3 = GhB.reduce (A, monoid) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.reduce (C1, monoid, A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.reduce (C1, A, monoid) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.reduce (c, monoid, A, desc)
%----------------------------------------------------------------------

% 2 matricies: c, A
% 1 string: monoid

C2 = sum (A, 'all') ;
C3 = GhB.reduce (A, monoid, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.reduce (C1, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.reduce (C1, A, monoid, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.reduce (c, accum, monoid, A)
%----------------------------------------------------------------------

% 2 matrices: c, A
% 2 strings: accum, monoid

C2 = C * sum (A, 'all') ;
C3 = GhB.reduce (C, accum, monoid, A) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.reduce (C1, accum, monoid, A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.reduce (C1, accum, A, monoid) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.reduce (C1, A, accum, monoid) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.reduce (accum, C1, monoid, A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.reduce (accum, C1, A, monoid) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.reduce (accum, monoid, C1, A) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.reduce (c, accum, monoid, A, desc)
%----------------------------------------------------------------------

% 2 matrices: c, A
% 2 strings: accum, monoid

C2 = C * sum (A, 'all') ;
C3 = GhB.reduce (C, accum, monoid, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.reduce (C1, accum, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.reduce (C1, accum, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.reduce (C1, A, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.reduce (accum, C1, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.reduce (accum, C1, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.reduce (accum, monoid, C1, A, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest143: all tests passed\n') ;

