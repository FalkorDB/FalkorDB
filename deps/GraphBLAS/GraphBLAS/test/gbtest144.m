function gbtest144
%GBTEST144 test GhB.vreduce
%
% GhB.vreduce (C, op, A)                   C = op (A)
% GhB.vreduce (C, accum, op, A)            C += op (A)
% GhB.vreduce (C, M, op, A)                C<M> = op (A)
% GhB.vreduce (C, M, accum, op, A)         C<M> += op (A)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C     = GhB.random (9, 1, 0.5, 'range', [-1 1]) ;
M     = GhB.random (9, 1, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GhB.random (9, 9, 0.5, 'range', [-1 1]) ;
desc  = struct ;
C0    = GhB (9, 1) ;

monoid = '+' ;

%----------------------------------------------------------------------
% GhB.vreduce (monoid, A)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string: monoid

C2 = sum (A,2) ;
C3 = GhB.vreduce (monoid, A) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.vreduce (C1, monoid, A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.vreduce (C1, A, monoid) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.vreduce (C, monoid, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string: monoid

C2 = sum (A,2) ;
C3 = GhB.vreduce (monoid, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.vreduce (C1, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.vreduce (C1, A, monoid, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.vreduce (C, accum, monoid, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 strings: accum, monoid

C2 = C + sum (A,2) ;
C3 = GhB.vreduce (C, accum, monoid, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.vreduce (C1, accum, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (C1, accum, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (C1, A, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (accum, monoid, C1, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.vreduce (C, M, monoid, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string: monoid

% C<M> = monoid (A)
T = sum (A,2) ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.vreduce (C, M, monoid, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.vreduce (C1, M, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (C1, monoid, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (monoid, C1, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (C1, M, A, monoid, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.vreduce (C, M, accum, monoid, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 strings: accum, monoid

% C<M> += monoid (A)

T = C + sum (A,2) ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.vreduce (C, M, accum, monoid, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.vreduce (C1, M, accum, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (C1, accum, monoid, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (C1, accum, M, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (C1, accum, M, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (C1, M, A, accum, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (C1, M, accum, A, monoid, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (accum, monoid, C1, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (accum, C1, monoid, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (accum, C1, M, monoid, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.vreduce (accum, C1, M, A, monoid, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest144: all tests passed\n') ;

