function gbtest142
%GBTEST142 test GhB.mxm (inplace usage)
%
% GhB.mxm (C, semiring, A, B)                  C = A*B
% GhB.mxm (C, accum, semiring, A, B)           C += A*B
% GhB.mxm (C, M, semiring, A, B)               C<M> = A*B
% GhB.mxm (C, M, accum, semiring, A, B)        C<M> += A*B

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C     = GhB.random (9, 9, 0.5) ;
M     = GhB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GhB.random (9, 9, 0.5) ;
B     = GhB.random (9, 9, 0.5) ;
desc  = struct ;
C0    = GhB (9, 9) ;

semiring = '+.*' ;

%----------------------------------------------------------------------
% GhB.mxm (C, semiring, A, B)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 1 string: semiring

C2 = A*B ;
C3 = GhB.mxm (semiring, A, B) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.mxm (C1, semiring, A, B) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.mxm (C1, A, semiring, B) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.mxm (C1, A, B, semiring) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.mxm (semiring, A, B, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 1 string: semiring

C2 = A*B ;
C3 = GhB.mxm (semiring, A, B, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.mxm (C1, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.mxm (C1, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.mxm (C1, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.mxm (C, accum, semiring, A, B, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 2 strings: accum, semiring

% C = accum (C, A*B) ;

C2 = C + A*B ;
C3 = GhB.mxm (C, accum, semiring, A, B, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.mxm (C1, accum, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, accum, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, accum, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, A, accum, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, A, accum, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, A, B, accum, semiring, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.mxm (C, M, semiring, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 1 string: semiring

% C<M> = A*B ;

T = A*B ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.mxm (C, M, semiring, A, B, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.mxm (semiring, C1, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, M, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, M, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, M, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.mxm (C, M, accum, semiring, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 2 string: accum, semiring

% C<M> = accum (C, A*B) ;

T = C + A*B ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.mxm (C, M, accum, semiring, A, B, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.mxm (C1, M, accum, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, M, accum, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, M, accum, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, accum, semiring, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, accum, M, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, accum, M, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, accum, M, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, M, A, B, accum, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, M, A, accum, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (C1, M, A, accum, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (accum, semiring, C1, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (accum, C1, semiring, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (accum, C1, M, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (accum, C1, M, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.mxm (accum, C1, M, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest142: all tests passed\n') ;

