function gbtest89 (ghb, ghb2)
%GBTEST89 test [GrB,GhB].extract
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

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
if (nargin < 2)
    ghb2 = ghb ;
end
gtb_name = gtb_prep (ghb) ;

C     = gtb_random (ghb2, 4, 3, 0.5) ;
M     = gtb_random (ghb2, 4, 3, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = gtb_random (ghb2, 9, 9, 0.5) ;
I     = { [1 4 2 5] } ;
J     = { [3 2 7 ] } ;
desc  = struct ;

Aij   = gtb_random (ghb2, 4, 3, 0.5) ;

V     = gtb_random (ghb2, 4, 1, 0.7) ;
W     = gtb_random (ghb2, 4, 1, 0.7, 'range', logical ([false true])) ;
U     = gtb_random (ghb2, 9, 1, 0.7) ;

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

C1 = gtb_extract (ghb, Aij) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, aij) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 0 strings:

C2 = Aij ;

C1 = gtb_extract (ghb, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (A, I, J)
%----------------------------------------------------------------------

% 1 matrix: A
% 2 indices: I, J
% 0 strings:

C2 = A (i,j) ;
c2 = a (i,j) ;
assert (isequal (c2, C2)) ;

C1 = gtb_extract (ghb, A, I, J) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, A, J) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, J, A) ; assert (isequal (C1, C2)) ;

C1 = gtb_extract (ghb, a, I, J) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, a, J) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, J, a) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (A, I, J, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 2 indices: I, J
% 0 strings:

C2 = A (i,j) ;
c2 = a (i,j) ;
assert (isequal (c2, C2)) ;

C1 = gtb_extract (ghb, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, J, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_extract (ghb, a, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, J, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (C, M, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 0 indices:
% 0 strings:

C2 = gtb (ghb, C) ;
C2 (M) = Aij (M) ;

c2 = c ;
c2 (m) = aij (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_extract (ghb, C, M, Aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (C, M, A, I, J, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices: I, J
% 0 strings:

% C<M> = A (I,J)

T = A (I,J) ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = a (i,j) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_extract (ghb, C, M, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, I, J, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_extract (ghb, C, I, J, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, M, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, M, A, J, desc) ; assert (isequal (C1, C2)) ;

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

C1 = gtb_extract (ghb, C, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, C, Aij, desc) ; assert (isequal (C1, C2)) ;

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

C1 = gtb_extract (ghb, C, accum, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, accum, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, accum, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, A, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, A, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, A, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, J, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, J, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, accum, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, accum, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, A, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, A, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, C, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, C, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, C, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, I, C, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, I, C, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, I, J, C, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, J, accum, C, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, J, accum, C, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, J, accum, C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_extract (ghb, c, accum, a, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, accum, I, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, accum, I, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, a, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, a, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, a, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, J, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, J, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, accum, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, accum, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, a, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, a, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, c, a, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, c, I, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, c, I, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, I, c, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, I, c, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, I, J, c, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, J, accum, c, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, J, accum, c, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, I, J, accum, c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (C, M, accum, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 0 indices:
% 1 string: accum

% C<M> += A

T = C + Aij ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = c + aij ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_extract (ghb, C, M, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, C, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, accum, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, Aij, accum, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_extract (ghb, c, m, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, accum, c, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, accum, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, aij, accum, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.extract (C, M, accum, A, I, J, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices: I, J
% 1 string: accum

% C<M> += A (I,J)

T = C + A (i,j) ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = c + a (i,j) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_extract (ghb, C, M, accum, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, accum, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, accum, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, A, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, A, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, A, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, I, J, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, I, J, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, I, A, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, I, A, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, I, accum, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, M, I, accum, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, J, M, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, J, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, J, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, M, J, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, M, J, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, M, A, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, M, A, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, M, accum, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, M, accum, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, accum, M, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, accum, M, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, I, accum, J, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, accum, M, A, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, accum, M, I, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, accum, M, I, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, accum, I, M, A, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, accum, I, M, J, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, C, accum, I, J, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_extract (ghb, c, m, accum, a, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, accum, I, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, accum, I, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, a, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, a, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, a, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, I, J, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, I, J, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, I, a, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, I, a, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, I, accum, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, m, I, accum, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, J, m, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, J, m, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, J, accum, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, m, J, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, m, J, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, m, a, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, m, a, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, m, accum, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, m, accum, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, accum, m, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, accum, m, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, I, accum, J, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, accum, m, a, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, accum, m, I, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, accum, m, I, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, accum, I, m, a, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, accum, I, m, J, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_extract (ghb, c, accum, I, J, m, a, desc) ; assert (isequal (C1, C2)) ;

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

V1 = gtb_extract (ghb, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, I, U, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_extract (ghb, u, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, I, u, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.extract (V, W, U, I, desc)
%----------------------------------------------------------------------

% 3 vectors: V, W, U
% 1 index: I
% 0 strings:

% V<W> = U(I)

T = U (i) ;
V2 = gtb (ghb, V) ;
V2 (W) = T (W) ;

t = u (i) ;
v2 = v ; 
v2 (w) = t (w) ;
assert (isequal (v2, V2)) ;

V1 = gtb_extract (ghb, V, W, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, W, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, I, W, U, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, I, V, W, U, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_extract (ghb, v, w, u, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, w, I, u, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, I, w, u, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, I, v, w, u, desc) ; assert (isequal (V1, V2)) ;

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

V1 = gtb_extract (ghb, V, accum, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, accum, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, U, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, U, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, accum, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, accum, I, U, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_extract (ghb, v, accum, u, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, accum, I, u, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, u, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, u, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, accum, u, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, accum, I, u, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.extract (V, W, accum, U, I, desc)
%----------------------------------------------------------------------

% 3 vectors: V, W, U
% 1 index: I
% 1 strings: accum

% V<W> += U(I)

T = V + U (i) ;
V2 = gtb (ghb, V) ;
V2 (W) = T (W) ;

t = v + u (i) ;
v2 = v ;
v2 (w) = t (w) ;
assert (isequal (v2, V2)) ;

V1 = gtb_extract (ghb, V, W, accum, U, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, W, accum, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, W, U, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, W, U, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, W, I, U, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, W, I, accum, U, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, I, W, U, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, I, W, accum, U, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, I, accum, W, U, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, accum, I, W, U, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, accum, W, I, U, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, V, accum, W, U, I, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_extract (ghb, v, w, accum, u, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, w, accum, I, u, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, w, u, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, w, u, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, w, I, u, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, w, I, accum, u, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, I, w, u, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, I, w, accum, u, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, I, accum, w, u, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, accum, I, w, u, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, accum, w, I, u, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_extract (ghb, v, accum, w, u, I, desc) ; assert (isequal (V1, V2)) ;

fprintf ('gbtest89 (%d): all tests passed\n', ghb) ;

