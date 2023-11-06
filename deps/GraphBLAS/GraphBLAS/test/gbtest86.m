function gbtest86
%GBTEST86 test GrB.mxm
%
% C = GrB.mxm (semiring, A, B)
% C = GrB.mxm (semiring, A, B, desc)
% C = GrB.mxm (C, accum, semiring, A, B, desc)
% C = GrB.mxm (C, M, semiring, A, B, desc)
% C = GrB.mxm (C, M, accum, semiring, A, B, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (9, 9, 0.5) ;
M     = GrB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GrB.random (9, 9, 0.5) ;
B     = GrB.random (9, 9, 0.5) ;
desc  = struct ;

semiring = '+.*' ;

c = double (C) ;
m = logical (M) ;
a = double (A) ;
b = double (B) ;

%----------------------------------------------------------------------
% C = GrB.mxm (semiring, A, B)
%----------------------------------------------------------------------

% 2 matrices: A, B
% 1 string: semiring

C2 = A*B ;
c2 = a*b ;
assert (isequal (c2, C2)) ;

C1 = GrB.mxm (semiring, A, B) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (A, semiring, B) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (A, B, semiring) ; assert (isequal (C1, C2)) ;

C1 = GrB.mxm (semiring, a, b) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (a, semiring, b) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (a, b, semiring) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.mxm (semiring, A, B, desc)
%----------------------------------------------------------------------

% 2 matrices: A, B
% 1 string: semiring

C2 = A*B ;
c2 = a*b ;
assert (isequal (c2, C2)) ;

C1 = GrB.mxm (semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (A, B, semiring, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.mxm (semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (a, b, semiring, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.mxm (C, accum, semiring, A, B, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 2 strings: accum, semiring

% C = accum (C, A*B) ;

C2 = C + A*B ;
c2 = c + a*b ;
assert (isequal (c2, C2)) ;

C1 = GrB.mxm (C, accum, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, accum, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, accum, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, A, accum, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, A, accum, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, A, B, accum, semiring, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.mxm (c, accum, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, accum, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, accum, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, a, accum, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, a, accum, b, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, a, b, accum, semiring, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.mxm (C, M, semiring, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 1 string: semiring

% C<M> = A*B ;

T = A*B ;
C2 = C ;
C2 (M) = T (M) ;

t = a*b ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.mxm (semiring, C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, M, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, M, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, M, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.mxm (semiring, c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, m, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, m, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, m, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.mxm (C, M, accum, semiring, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 2 string: accum, semiring

% C<M> = accum (C, A*B) ;

T = C + A*B ;
C2 = C ;
C2 (M) = T (M) ;

t = c + a*b ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.mxm (C, M, accum, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, M, accum, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, M, accum, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, accum, semiring, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, accum, M, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, accum, M, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, accum, M, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, M, A, B, accum, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, M, A, accum, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (C, M, A, accum, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (accum, semiring, C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (accum, C, semiring, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (accum, C, M, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (accum, C, M, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (accum, C, M, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.mxm (c, m, accum, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, m, accum, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, m, accum, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, accum, semiring, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, accum, m, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, accum, m, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, accum, m, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, m, a, b, accum, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, m, a, accum, b, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (c, m, a, accum, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (accum, semiring, c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (accum, c, semiring, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (accum, c, m, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (accum, c, m, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.mxm (accum, c, m, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest86: all tests passed\n') ;

