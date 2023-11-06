function gbtest85
%GBTEST85 test GrB.subassign
%
% C = GrB.subassign (C, A) ;
% C = GrB.subassign (C, M, A) ;
% C = GrB.subassign (C, accum, A) ;
% C = GrB.subassign (C, M, accum, A) ;
%
% V = GrB.subassign (V, U, I) ;
% V = GrB.subassign (V, W, U, I) ;
% V = GrB.subassign (V, accum, U, I) ;
% V = GrB.subassign (V, W, accum, U, I) ;
%
% C = GrB.subassign (C, A, I, J) ;
% C = GrB.subassign (C, M, A, I, J) ;
% C = GrB.subassign (C, accum, A, I, J) ;
% C = GrB.subassign (C, M, accum, A, I, J) ;

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (9, 9, 0.5) ;
M     = GrB.random (9, 9, 0.5, 'range', logical ([false true])) ;
Mij   = GrB.random (4, 3, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GrB.random (9, 9, 0.5) ;
I     = { [1 4 2 5] } ;
J     = { [3 2 7 ] } ;
desc  = struct ;

Aij   = GrB.random (4, 3, 0.5) ;
V     = GrB.random (9, 1, 0.7) ;
Wi    = GrB.random (4, 1, 0.7, 'range', logical ([false true])) ;
Ui    = GrB.random (4, 1, 0.7) ;

c = double (C) ;
a = double (A) ;
m = logical (M) ;
i = I {1} ;
j = J {1} ;

aij = double (Aij) ;

mij = logical (Mij) ;

v = double (V) ;

wi  = logical (Wi) ;
ui  = double (Ui) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, A) ;
%----------------------------------------------------------------------

% 1 matrix: A
% 0 strings:

C2 = A ;

C1 = GrB.subassign (C, A) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, a) ; assert (isequal (C1, C2)) ;

C1 = GrB.subassign (C, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, M, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, A
% 0 strings:

C2 = C ;
C2 (M) = A (M) ;

c2 = c ;
c2 (m) = a (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.subassign (C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, accum, A) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string:   accum

% C += A

C2 = C + A ;

c2 = c + a ;
assert (isequal (c2, C2)) ;

C1 = GrB.subassign (C, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, M, accum, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string:   accum

% C<M> += A
C2 = C ;
C2 (M) = C2 (M) + A (M) ;

c2 = c ;
c2 (m) = c2 (m) + a (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.subassign (C, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, M, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.subassign (c, m, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, m, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, accum, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% V = GrB.subassign (V, U, I) ;
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

V1 = GrB.subassign (V, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, V, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.subassign (v, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, v, ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.subassign (V, W, U, I) ;
%----------------------------------------------------------------------

% 3 vectors: V, Wi, Ui
% 0 strings:
% 1 index:   I

% V(I)<W> = Ui

S = V (i) ;
% with accum
% T = S + Ui ;
% with no accum:
T = Ui ;
% with mask:
S (Wi) = T (Wi) ;
% with no mask:
% S = T ;
V2 = V ;
V2 (i) = S ;

s = v (i) ;
t = ui ;
s (wi) = t (wi) ;
v2 = v ;
v2 (i) = s ;
assert (isequal (v2, V2)) ;

V1 = GrB.subassign (V, Wi, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, Wi, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, I, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, V, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.subassign (v, wi, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, wi, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, I, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, v, wi, ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.subassign (V, accum, U, I) ;
%----------------------------------------------------------------------

% 2 vectors: V, Ui
% 1 string:  accum
% 1 index:   I

% V(I)<W> = accum (V(I), Ui)

S = V (i) ;
% with accum
T = S + Ui ;
% with no accum:
% T = Ui ;
% with mask:
% S (Wi) = T (Wi) ;
% with no mask:
S = T ;
V2 = V ;
V2 (i) = S ;

V1 = GrB.subassign (V, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, V, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, V, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, I, V, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, accum, V, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, V, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, V, Ui, accum, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.subassign (v, accum, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, accum, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, I, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, I, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, v, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, v, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, I, v, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, accum, v, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, v, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, v, ui, accum, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% V = GrB.subassign (V, W, accum, U, I) ;
%----------------------------------------------------------------------

% 3 vectors: V, W, Ui
% 1 string:  accum
% 1 index:   I

% V(I)<W> = accum (V(I), Ui)

S = V (i) ;
% with accum
T = S + Ui ;
% with no accum:
% T = Ui ;
% with mask:
S (Wi) = T (Wi) ;
% with no mask:
% S = T ;
V2 = V ;
V2 (i) = S ;

s = v (i) ;
t = s + ui ;
s (wi) = t (wi) ;
v2 = v ;
v2 (i) = s ;
assert (isequal (v2, V2)) ;

V1 = GrB.subassign (V, Wi, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, Wi, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, Wi, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, Wi, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, Wi, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, Wi, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, accum, Wi, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, accum, Wi, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, accum, I, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, I, Wi, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, I, Wi, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (V, I, accum, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, V, Wi, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, V, Wi, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, V, I, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, I, V, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, V, Wi, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, V, Wi, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, V, accum, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, accum, V, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = GrB.subassign (v, wi, accum, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, wi, accum, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, wi, ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, wi, ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, wi, I, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, wi, I, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, accum, wi, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, accum, wi, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, accum, I, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, I, wi, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, I, wi, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (v, I, accum, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, v, wi, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, v, wi, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, v, I, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (accum, I, v, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, v, wi, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, v, wi, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, v, accum, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GrB.subassign (I, accum, v, wi, ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, A, I, J) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 indices:  I, J

% C(I,J)<M> = accum (C(I,J), Aij)

% S = C (i,j) ;
% with accum:
% T = S + Aij ;
% with no accum:
T = Aij ;
% with mask:
% S (Mij) = T (Mij) ;
% with no mask:
S = T ;
C2 = C ;
C2 (i,j) = S ;

t = aij ;
s = t ;
c2 = c ;
c2 (i,j) = s ;
assert (isequal (c2, C2)) ;

C1 = GrB.subassign (C, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, C, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Aij, J, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.subassign (c, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, c, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, aij, J, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, M, A, I, J) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices:  I, J

% C(I,J)<M> = accum (C(I,J), Aij)

S = C (i,j) ;
% with accum:
% T = S + Aij ;
% with no accum:
T = Aij ;
% with mask:
S (Mij) = T (Mij) ;
% with no mask:
% S = T ;
C2 = C ;
C2 (i,j) = S ;

s = c (i,j) ;
t = aij ;
s (mij) = t (mij) ;
c2 = c ;
c2 (i,j) = s ;
assert (isequal (c2, C2)) ;

C1 = GrB.subassign (C, Mij, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, C, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.subassign (c, mij, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, c, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, accum, A, I, J) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 indices:  I, J
% 1 string:   accum

% C(I,J)<M> = accum (C(I,J), Aij)

S = C (i,j) ;
% with accum:
T = S + Aij ;
% with no accum:
% T = Aij ;
% with mask:
% S (Mij) = T (Mij) ;
% with no mask:
S = T ;
C2 = C ;
C2 (i,j) = S ;

s = c (i,j) ;
t = s + aij ;
s = t ;
c2 = c ;
c2 (i,j) = s ;
assert (isequal (c2, C2)) ;

C1 = GrB.subassign (C, accum, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, accum, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, accum, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, accum, C, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, C, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, C, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, C, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, C, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, J, C, Aij, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.subassign (c, accum, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, accum, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, accum, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, accum, c, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, c, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, c, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, c, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, c, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, c, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, c, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, c, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, J, c, aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, M, accum, A, I, J) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices:  I, J
% 1 string:   accum

% C(I,J)<M> = accum (C(I,J), Aij)

S = C (i,j) ;
% with accum:
T = S + Aij ;
% with no accum:
% T = Aij ;
% with mask:
S (Mij) = T (Mij) ;
% with no mask:
% S = T ;
C2 = C ;
C2 (i,j) = S ;

s = c (i,j) ;
t = s + aij ;
s (mij) = t (mij) ;
c2 = c ;
c2 (i,j) = s ;
assert (isequal (c2, C2)) ;

C1 = GrB.subassign (C, Mij, accum, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, accum, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, accum, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, Aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, Aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, Aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, I, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, I, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, I, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, I, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, I, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, Mij, I, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, accum, Mij, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, accum, Mij, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, accum, Mij, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, accum, I, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, accum, I, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, accum, I, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, J, accum, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, J, Mij, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, J, Mij, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, accum, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, accum, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, accum, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Mij, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Mij, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Mij, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Mij, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Mij, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (C, I, Mij, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, Mij, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, Mij, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, Mij, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, I, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, I, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, C, I, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, J, C, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, C, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, C, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, C, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, accum, C, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, C, accum, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, C, Mij, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, C, Mij, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, accum, J, C, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, accum, C, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, accum, C, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, accum, C, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, accum, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, accum, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, accum, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, J, accum, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, J, Mij, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, J, Mij, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Mij, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Mij, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Mij, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Mij, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Mij, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, C, Mij, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.subassign (c, mij, accum, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, accum, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, accum, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, I, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, I, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, I, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, I, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, I, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, mij, I, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, accum, mij, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, accum, mij, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, accum, mij, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, accum, I, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, accum, I, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, accum, I, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, J, accum, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, J, mij, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, J, mij, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, accum, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, accum, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, accum, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, mij, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, mij, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, mij, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, mij, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, mij, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (c, I, mij, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, c, mij, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, c, mij, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, c, mij, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, c, I, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, c, I, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, c, I, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, J, c, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, c, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, c, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (accum, I, c, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, accum, c, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, c, accum, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, c, mij, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, J, c, mij, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, accum, J, c, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, accum, c, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, accum, c, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, accum, c, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, accum, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, accum, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, accum, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, J, accum, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, J, mij, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, J, mij, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, mij, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, mij, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, mij, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, mij, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, mij, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.subassign (I, c, mij, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest85: all tests passed\n') ;

