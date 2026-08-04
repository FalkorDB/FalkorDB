function gbtest147
%GBTEST147 test GhB.extract (inplace usage)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C     = GhB.random (4, 3, 0.5) ;
M     = GhB.random (4, 3, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GhB.random (9, 9, 0.5) ;
I     = { [1 4 2 5] } ;
J     = { [3 2 7 ] } ;
desc  = struct ;
C0    = GhB (4, 3) ;

Aij   = GhB.random (4, 3, 0.5) ;

V     = GhB.random (4, 1, 0.7) ;
W     = GhB.random (4, 1, 0.7, 'range', logical ([false true])) ;
U     = GhB.random (9, 1, 0.7) ;
V0    = GhB (4, 1) ;

i = I {1} ;
j = J {1} ;

%----------------------------------------------------------------------
% GhB.extract (C, A)
%----------------------------------------------------------------------

% 1 matrix: A
% 0 strings:

C2 = GhB (Aij) ;
C3 = GhB.extract (Aij) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.extract (C1, Aij) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.extract (A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 0 strings:

C2 = GhB (Aij) ;
C3 = GhB.extract (Aij, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.extract (C1, Aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.extract (C, A, I, J)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 indices: I, J
% 0 strings:

C2 = A (i,j) ;
C3 = GhB.extract (A, I, J) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.extract (C1, A, I, J) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.extract (C1, I, A, J) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.extract (C1, I, J, A) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.extract (C, A, I, J, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 indices: I, J
% 0 strings:

C2 = A (i,j) ;
C3 = GhB.extract (A, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.extract (C1, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.extract (C1, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.extract (C1, I, J, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.extract (C, M, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 0 indices:
% 0 strings:

C2 = GhB (C) ;
C2 (M) = Aij (M) ;

C3 = GhB.extract (C, M, Aij, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.extract (C1, M, Aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.extract (C, M, A, I, J, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices: I, J
% 0 strings:

% C<M> = A (I,J)

T = A (I,J) ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.extract (C, M, A, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.extract (C1, M, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, I, J, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GhB (C) ; GhB.extract (C1, I, J, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, M, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, M, A, J, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.extract (C, accum, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 0 indices:
% 1 string: accum

% C += A

C2 = C + Aij ;
C3 = GhB.extract (C, accum, Aij, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.extract (C1, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (accum, C1, Aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.extract (C, accum, A, I, J, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 indices: I, J
% 1 string: accum

% C += A (i,j)

C2 = C + A (i,j) ;
C3 = GhB.extract (C, accum, A, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.extract (C1, accum, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, accum, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, accum, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, A, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, A, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, A, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, J, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, J, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, accum, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, accum, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, A, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, A, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (accum, C1, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (accum, C1, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (accum, C1, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (accum, I, C1, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (accum, I, C1, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (accum, I, J, C1, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (I, J, accum, C1, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (I, J, accum, C1, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (I, J, accum, C1, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.extract (C, M, accum, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 0 indices:
% 1 string: accum

% C<M> += A

T = C + Aij ;
C2 = GhB (C) ;
C2 (M) = T (M) ;

C1 = GhB (C) ; GhB.extract (C1, M, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (accum, C1, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, accum, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, Aij, accum, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.extract (C, M, accum, A, I, J, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices: I, J
% 1 string: accum

% C<M> += A (I,J)

T = C + A (i,j) ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.extract (C, M, accum, A, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.extract (C1, M, accum, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, accum, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, accum, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, A, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, A, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, A, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, I, J, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, I, J, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, I, A, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, I, A, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, I, accum, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, M, I, accum, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, J, M, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, J, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, J, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, M, J, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, M, J, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, M, A, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, M, A, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, M, accum, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, M, accum, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, accum, M, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, accum, M, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, I, accum, J, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, accum, M, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, accum, M, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, accum, M, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, accum, I, M, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, accum, I, M, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.extract (C1, accum, I, J, M, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.extract (V, U, I, desc)
%----------------------------------------------------------------------

% 2 vectors: V, U
% 1 index: I
% 0 strings:

% V = U(I)

V2 = U (i) ;
V3 = GhB.extract (U, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V0) ; GhB.extract (V1, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V0) ; GhB.extract (V1, I, U, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.extract (V, W, U, I, desc)
%----------------------------------------------------------------------

% 3 vectors: V, W, U
% 1 index: I
% 0 strings:

% V<W> = U(I)

T = U (i) ;
V2 = GhB (V) ;
V2 (W) = T (W) ;
V3 = GhB.extract (V, W, U, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.extract (V1, W, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, W, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, I, W, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (I, V1, W, U, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.extract (V, accum, U, I, desc)
%----------------------------------------------------------------------

% 2 vectors: V, U
% 1 index: I
% 1 string: accum

% V += U(I)

V2 = V + U (i) ;
V3 = GhB.extract (V, accum, U, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.extract (V1, accum, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, accum, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, U, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, U, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, accum, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, accum, I, U, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.extract (V, W, accum, U, I, desc)
%----------------------------------------------------------------------

% 3 vectors: V, W, U
% 1 index: I
% 1 strings: accum

% V<W> += U(I)

T = V + U (i) ;
V2 = GhB (V) ;
V2 (W) = T (W) ;
V3 = GhB.extract (V, W, accum, U, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.extract (V1, W, accum, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, W, accum, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, W, U, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, W, U, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, W, I, U, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, W, I, accum, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, I, W, U, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, I, W, accum, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, I, accum, W, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, accum, I, W, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, accum, W, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.extract (V1, accum, W, U, I, desc) ; assert (isequal (V1, V2)) ;

fprintf ('gbtest147: all tests passed\n') ;

