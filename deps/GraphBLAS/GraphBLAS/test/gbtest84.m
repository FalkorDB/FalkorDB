function gbtest84
%GBTEST84 test GrB.assign
%
% C = GrB.assign (C, A) ;
% C = GrB.assign (C, M, A) ;
% C = GrB.assign (C, accum, A) ;
% C = GrB.assign (C, M, accum, A) ;
%
% V = GrB.assign (V, U, I) ;
% V = GrB.assign (V, W, U, I) ;
% V = GrB.assign (V, accum, U, I) ;
% V = GrB.assign (V, W, accum, U, I) ;
%
% C = GrB.assign (C, A, I, J) ;
% C = GrB.assign (C, M, A, I, J) ;
% C = GrB.assign (C, accum, A, I, J) ;
% C = GrB.assign (C, M, accum, A, I, J) ;

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (9, 9, 0.5) ;
M     = GrB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GrB.random (9, 9, 0.5) ;
I     = { [1 4 2 5] } ;
J     = { [3 2 7 ] } ;
desc  = struct ;

Aij   = GrB.random (4, 3, 0.5) ;

V     = GrB.random (9, 1, 0.7) ;
W     = GrB.random (9, 1, 0.7, 'range', logical ([false true])) ;

Ui    = GrB.random (4, 1, 0.7) ;

c = double (C) ;
a = double (A) ;
m = logical (M) ;
i = I {1} ;
j = J {1} ;

aij = double (Aij) ;

v = double (V) ;
w = logical (W) ;

ui  = double (Ui) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, A) ;
%----------------------------------------------------------------------

% 1 matrix: A
% 0 strings:

C2 = A ;

C1 = GrB.assign (C, A) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, a) ; assert (isequal (C1, C2)) ;

C1 = GrB.assign (C, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, M, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, A
% 0 strings:

C2 = C ;
C2 (M) = A (M) ;

c2 = c ;
c2 (m) = a (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.assign (C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, accum, A) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string:   accum

% C += A

C2 = C + A ;

c2 = c + a ;
assert (isequal (c2, C2)) ;

C1 = GrB.assign (C, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, M, accum, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string:   accum

% C<M> += A
C2 = C ;
C2 (M) = C2 (M) + A (M) ;

c2 = c ;
c2 (m) = c2 (m) + a (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.assign (C, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.assign (c, m, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, accum, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% V = GrB.assign (V, U, I) ;
%----------------------------------------------------------------------

% 2 vectors: V, U
% 0 strings:
% 1 index:   I

% V(I) = U

V2 = V ;
V2 (i) = Ui ;

v2 = v ;
v2 (i) = ui ;
assert (isequal (v2, V2)) ;

V1 = GrB.assign (V, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, V, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.assign (v, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, v, ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.assign (V, W, U, I) ;
%----------------------------------------------------------------------

% 3 vectors: V, W, Ui
% 0 strings:
% 1 index:   I

% V<W>(I) = Ui

% S = V (i) ;
% with accum
% S = S + Ui ;
% with no accum:
S = Ui ;
Z = V ;
Z (i) = S ;
% with mask:
V2 = V ;
V2 (W) = Z (W) ;
% with no mask:
% V2 = Z ;

% s = v (i) ;
s = ui ;
z = v ;
z (i) = s ;
v2 = v ;
v2 (w) = z (w) ;
assert (isequal (v2, V2)) ;

V1 = GrB.assign (V, W, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, W, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, I, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, V, W, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.assign (v, w, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, w, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, I, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, v, w, ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.assign (V, accum, U, I) ;
%----------------------------------------------------------------------

% 2 vectors: V, Ui
% 1 string:  accum
% 1 index:   I

% V<W>(I) = accum (V(I), Ui)

S = V (i) ;
% with accum:
S = S + Ui ;
% with no accum:
% S = Ui ;
Z = V ;
Z (i) = S ;
% with mask:
% V2 = V ;
% V2 (W) = Z (W) ;
% with no mask:
V2 = Z ;

s = v (i) ;
s = s + ui ;
z = v ;
z (i) = s ;
v2 = z ;
assert (isequal (v2, V2)) ;

V1 = GrB.assign (V, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, V, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, V, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, I, V, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, accum, V, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, V, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, V, Ui, accum, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.assign (v, accum, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, accum, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, I, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, I, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, v, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, v, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, I, v, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, accum, v, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, v, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, v, ui, accum, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.assign (V, W, accum, U, I) ;
%----------------------------------------------------------------------

% 3 vectors: V, W, Ui
% 1 string:  accum
% 1 index:   I

% V<W>(I) = accum (V(I), Ui)

S = V (i) ;
% with accum:
S = S + Ui ;
% with no accum:
% S = Ui ;
Z = V ;
Z (i) = S ;
% with mask:
V2 = V ;
V2 (W) = Z (W) ;
% with no mask:
% V2 = Z ;

s = v (i) ;
s = s + ui ;
z = v ;
z (i) = s ;
v2 = v ;
v2 (w) = z (w) ;
assert (isequal (v2, V2)) ;

V1 = GrB.assign (V, W, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, W, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, W, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, W, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, W, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, W, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, accum, W, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, accum, W, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, accum, I, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, I, W, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, I, W, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (V, I, accum, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, V, W, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, V, W, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, V, I, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, I, V, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, V, W, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, V, W, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, V, accum, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, accum, V, W, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.assign (v, w, accum, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, w, accum, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, w, ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, w, ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, w, I, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, w, I, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, accum, w, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, accum, w, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, accum, I, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, I, w, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, I, w, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (v, I, accum, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, v, w, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, v, w, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, v, I, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (accum, I, v, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, v, w, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, v, w, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, v, accum, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.assign (I, accum, v, w, ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, A, I, J) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 indices:  I, J

% C<M>(I,J) = accum (C(I,J), Aij)

% S = C (i,j) ;
% with accum:
% S = S + Aij ;
% with no accum:
S = Aij ;
Z = C ;
Z (i,j) = S ;
% with mask:
% C2 = C ;
% C2 (M) = Z (M) ;
% with no mask:
C2 = Z ;

% s = c (i,j) ;
s = aij ;
z = c ;
z (i,j) = s ;
c2 = z ;
assert (isequal (c2, C2)) ;

C1 = GrB.assign (C, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, C, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, Aij, J, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.assign (c, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, c, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, aij, J, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, M, A, I, J) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices:  I, J

% C<M>(I,J) = accum (C(I,J), Aij)

% S = C (i,j) ;
% with accum:
% S = S + Aij ;
% with no accum:
S = Aij ;
Z = C ;
Z (i,j) = S ;
% with mask:
C2 = C ;
C2 (M) = Z (M) ;
% with no mask:
% C2 = Z ;

% s = c (i,j) ;
s = aij ;
z = c ;
z (i,j) = s ;
c2 = c ;
c2 (m) = z (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.assign (C, M, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, C, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.assign (c, m, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, J, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, m, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, m, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, c, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, J, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, m, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, m, aij, J, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, accum, A, I, J) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 indices:  I, J
% 1 string:   accum

% C<M>(I,J) = accum (C(I,J), Aij)

S = C (i,j) ;
% with accum:
S = S + Aij ;
% with no accum:
% S = Aij ;
Z = C ;
Z (i,j) = S ;
% with mask:
% C2 = C ;
% C2 (M) = Z (M) ;
% with no mask:
C2 = Z ;

s = c (i,j) ;
s = s + aij ;
z = c ;
z (i,j) = s ;
c2 = z ;
assert (isequal (c2, C2)) ;

C1 = GrB.assign (C, accum, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, accum, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, accum, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, Aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, Aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, Aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, accum, C, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, C, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, C, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, C, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, C, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, J, C, Aij, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.assign (c, accum, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, accum, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, accum, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, accum, c, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, c, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, c, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, c, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, c, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, c, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, c, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, c, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, J, c, aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, M, accum, A, I, J) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices:  I, J
% 1 string:   accum

% C<M>(I,J) = accum (C(I,J), Aij)

S = C (i,j) ;
% with accum:
S = S + Aij ;
% with no accum:
% S = Aij ;
Z = C ;
Z (i,j) = S ;
% with mask:
C2 = C ;
C2 (M) = Z (M) ;
% with no mask:
% C2 = Z ;

s = c (i,j) ;
s = s + aij ;
z = c ;
z (i,j) = s ;
c2 = c ;
c2 (m) = z (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.assign (C, M, accum, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, accum, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, accum, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, Aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, Aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, Aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, I, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, I, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, I, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, I, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, I, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, M, I, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, accum, M, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, accum, M, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, accum, M, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, accum, I, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, accum, I, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, accum, I, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, J, accum, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, J, M, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, J, M, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, accum, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, accum, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, accum, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, M, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, M, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, M, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, M, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, M, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (C, I, M, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, M, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, M, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, M, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, I, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, I, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, C, I, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, J, C, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, C, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, C, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, C, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, accum, C, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, C, accum, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, C, M, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, C, M, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, accum, J, C, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, accum, C, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, accum, C, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, accum, C, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, accum, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, accum, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, accum, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, J, accum, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, J, M, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, J, M, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, M, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, M, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, M, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, M, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, M, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, C, M, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.assign (c, m, accum, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, accum, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, accum, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, I, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, I, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, I, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, I, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, I, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, m, I, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, accum, m, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, accum, m, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, accum, m, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, accum, I, J, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, accum, I, m, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, accum, I, m, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, J, accum, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, J, m, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, J, m, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, accum, J, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, accum, m, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, accum, m, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, m, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, m, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, m, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, m, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, m, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (c, I, m, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, c, m, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, c, m, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, c, m, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, c, I, m, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, c, I, m, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, c, I, J, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, J, c, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, c, J, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, c, m, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (accum, I, c, m, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, accum, c, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, c, accum, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, c, m, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, J, c, m, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, accum, J, c, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, accum, c, J, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, accum, c, m, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, accum, c, m, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, accum, J, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, accum, m, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, accum, m, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, J, accum, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, J, m, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, J, m, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, m, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, m, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, m, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, m, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, m, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.assign (I, c, m, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest84: all tests passed\n') ;
