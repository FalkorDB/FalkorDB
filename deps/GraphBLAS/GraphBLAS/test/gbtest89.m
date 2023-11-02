function gbtest89
%GBTEST89 test GrB.extract
%
% C = GrB.extract (A, desc)
% C = GrB.extract (A, I, J, desc)
% C = GrB.extract (C, M, A, desc)
% C = GrB.extract (C, M, A, I, J, desc)
% C = GrB.extract (C, accum, A, desc)
% C = GrB.extract (C, accum, A, I, J, desc)
% C = GrB.extract (C, M, accum, A, desc)
% C = GrB.extract (C, M, accum, A, I, J, desc)
%
% V = GrB.extract (U, I, desc)
% V = GrB.extract (V, W, U, I, desc)
% V = GrB.extract (V, accum, U, I, desc)
% V = GrB.extract (V, W, accum, U, I, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (4, 3, 0.5) ;
M     = GrB.random (4, 3, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GrB.random (9, 9, 0.5) ;
I     = { [1 4 2 5] } ;
J     = { [3 2 7 ] } ;
desc  = struct ;

Aij   = GrB.random (4, 3, 0.5) ;

V     = GrB.random (4, 1, 0.7) ;
W     = GrB.random (4, 1, 0.7, 'range', logical ([false true])) ;
U     = GrB.random (9, 1, 0.7) ;

c = double (C) ;
a = double (A) ;
m = logical (M) ;
i = I {1} ;
j = J {1} ;

aij = double (Aij) ;

v = double (V) ;
w = logical (W) ;
u = double (U) ;

%----------------------------------------------------------------------
% C = GrB.extract (A)
%----------------------------------------------------------------------

% 1 matrix: A
% 0 strings:

C2 = Aij ;

C1 = GrB.extract (Aij) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (aij) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 0 strings:

C2 = Aij ;

C1 = GrB.extract (Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (A, I, J)
%----------------------------------------------------------------------

% 1 matrix: A
% 2 indices: I, J
% 0 strings:

C2 = A (i,j) ;
c2 = a (i,j) ;
assert (isequal (c2, C2)) ;

C1 = GrB.extract (A, I, J) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, A, J) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, J, A) ; assert (isequal (C1, C2)) ;

C1 = GrB.extract (a, I, J) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, a, J) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, J, a) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (A, I, J, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 2 indices: I, J
% 0 strings:

C2 = A (i,j) ;
c2 = a (i,j) ;
assert (isequal (c2, C2)) ;

C1 = GrB.extract (A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, J, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.extract (a, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, J, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (C, M, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 0 indices:
% 0 strings:

C2 = C ;
C2 (M) = Aij (M) ;

c2 = c ;
c2 (m) = aij (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.extract (C, M, Aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (C, M, A, I, J, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices: I, J
% 0 strings:

% C<M> = A (I,J)

T = A (I,J) ;
C2 = C ;
C2 (M) = T (M) ;

t = a (i,j) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.extract (C, M, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, I, J, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.extract (C, I, J, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, M, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, M, A, J, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (C, accum, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 0 indices:
% 1 string: accum

% C += A

C2 = C + Aij ;

c2 = c + aij ;
assert (isequal (c2, C2)) ;

C1 = GrB.extract (C, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, C, Aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (C, accum, A, I, J, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 indices: I, J
% 1 string: accum

% C += A (i,j)

C2 = C + A (i,j) ;

c2 = c + a (i,j) ;
assert (isequal (c2, C2)) ;

C1 = GrB.extract (C, accum, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, accum, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, accum, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, A, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, A, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, A, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, J, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, J, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, accum, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, accum, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, A, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, A, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, C, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, C, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, C, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, I, C, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, I, C, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, I, J, C, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, J, accum, C, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, J, accum, C, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, J, accum, C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.extract (c, accum, a, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, accum, I, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, accum, I, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, a, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, a, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, a, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, J, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, J, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, accum, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, accum, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, a, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, a, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, c, a, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, c, I, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, c, I, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, I, c, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, I, c, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, I, J, c, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, J, accum, c, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, J, accum, c, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (I, J, accum, c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (C, M, accum, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 0 indices:
% 1 string: accum

% C<M> += A

T = C + Aij ;
C2 = C ;
C2 (M) = T (M) ;

t = c + aij ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.extract (C, M, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, C, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, accum, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, Aij, accum, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.extract (c, m, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (accum, c, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, accum, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, aij, accum, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (C, M, accum, A, I, J, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices: I, J
% 1 string: accum

% C<M> += A (I,J)

T = C + A (i,j) ;
C2 = C ;
C2 (M) = T (M) ;

t = c + a (i,j) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.extract (C, M, accum, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, accum, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, accum, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, A, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, A, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, A, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, I, J, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, I, J, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, I, A, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, I, A, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, I, accum, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, M, I, accum, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, J, M, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, J, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, J, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, M, J, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, M, J, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, M, A, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, M, A, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, M, accum, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, M, accum, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, accum, M, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, accum, M, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, I, accum, J, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, accum, M, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, accum, M, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, accum, M, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, accum, I, M, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, accum, I, M, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (C, accum, I, J, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.extract (c, m, accum, a, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, accum, I, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, accum, I, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, a, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, a, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, a, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, I, J, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, I, J, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, I, a, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, I, a, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, I, accum, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, m, I, accum, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, J, m, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, J, m, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, J, accum, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, m, J, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, m, J, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, m, a, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, m, a, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, m, accum, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, m, accum, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, accum, m, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, accum, m, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, I, accum, J, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, accum, m, a, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, accum, m, I, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, accum, m, I, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, accum, I, m, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, accum, I, m, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.extract (c, accum, I, J, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% V = GrB.extract (U, I, desc)
%----------------------------------------------------------------------

% 1 vector: V
% 1 index: I
% 0 strings:

% V = U(I)

V2 = U (i) ;

v2 = u (i) ;
assert (isequal (v2, V2)) ;

V1 = GrB.extract (U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (I, U, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.extract (u, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (I, u, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.extract (V, W, U, I, desc)
%----------------------------------------------------------------------

% 3 vectors: V, W, U
% 1 index: I
% 0 strings:

% V<W> = U(I)

T = U (i) ;
V2 = V ;
V2 (W) = T (W) ;

t = u (i) ;
v2 = v ; 
v2 (w) = t (w) ;
assert (isequal (v2, V2)) ;

V1 = GrB.extract (V, W, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, W, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, I, W, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (I, V, W, U, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.extract (v, w, u, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, w, I, u, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, I, w, u, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (I, v, w, u, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.extract (V, accum, U, I, desc)
%----------------------------------------------------------------------

% 2 vectors: V, U
% 1 index: I
% 1 string: accum

% V += U(I)

V2 = V + U (i) ;

v2 = v + u (i) ;
assert (isequal (v2, V2)) ;

V1 = GrB.extract (V, accum, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, accum, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, U, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, U, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, accum, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, accum, I, U, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.extract (v, accum, u, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, accum, I, u, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, u, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, u, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, accum, u, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, accum, I, u, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.extract (V, W, accum, U, I, desc)
%----------------------------------------------------------------------

% 3 vectors: V, W, U
% 1 index: I
% 1 strings: accum

% V<W> += U(I)

T = V + U (i) ;
V2 = V ;
V2 (W) = T (W) ;

t = v + u (i) ;
v2 = v ;
v2 (w) = t (w) ;
assert (isequal (v2, V2)) ;

V1 = GrB.extract (V, W, accum, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, W, accum, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, W, U, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, W, U, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, W, I, U, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, W, I, accum, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, I, W, U, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, I, W, accum, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, I, accum, W, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, accum, I, W, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, accum, W, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (V, accum, W, U, I, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.extract (v, w, accum, u, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, w, accum, I, u, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, w, u, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, w, u, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, w, I, u, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, w, I, accum, u, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, I, w, u, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, I, w, accum, u, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, I, accum, w, u, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, accum, I, w, u, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, accum, w, I, u, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.extract (v, accum, w, u, I, desc) ; assert (isequal (V1, V2)) ;

fprintf ('gbtest89: all tests passed\n') ;

