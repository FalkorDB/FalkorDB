function gbtest85 (ghb)
%GBTEST85 test [GrB,GhB].subassign
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

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

C     = gtb_random (ghb, 9, 9, 0.5) ;
M     = gtb_random (ghb, 9, 9, 0.5, 'range', logical ([false true])) ;
Mij   = gtb_random (ghb, 4, 3, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = gtb_random (ghb, 9, 9, 0.5) ;
I     = { [1 4 2 5] } ;
J     = { [3 2 7 ] } ;
desc  = struct ;

Aij   = gtb_random (ghb, 4, 3, 0.5) ;
V     = gtb_random (ghb, 9, 1, 0.7) ;
Wi    = gtb_random (ghb, 4, 1, 0.7, 'range', logical ([false true])) ;
Ui    = gtb_random (ghb, 4, 1, 0.7) ;

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

C2 = gtb (ghb, A) ;

C1 = gtb_subassign (ghb, C, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, a) ; assert (isequal (C1, C2)) ;

C1 = gtb_subassign (ghb, C, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, M, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, A
% 0 strings:

C2 = gtb (ghb, C) ;
C2 (M) = A (M) ;

c2 = c ;
c2 (m) = a (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_subassign (ghb, C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, accum, A) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string:   accum

% C += A

C2 = C + A ;

c2 = c + a ;
assert (isequal (c2, C2)) ;

C1 = gtb_subassign (ghb, C, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.subassign (C, M, accum, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string:   accum

% C<M> += A
C2 = gtb (ghb, C) ;
C2 (M) = C2 (M) + A (M) ;

c2 = c ;
c2 (m) = c2 (m) + a (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_subassign (ghb, C, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, M, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_subassign (ghb, c, m, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, m, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, accum, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% V = GrB.subassign (V, U, I) ;
%----------------------------------------------------------------------

% 2 vectors: V, U
% 0 strings:
% 1 index:   I

% V(I) = U

V2 = gtb (ghb, V) ;
V2 (i) = Ui ;

v2 = v ;
v2 (i) = ui ;
assert (isequal (v2, V2)) ;

V1 = gtb_subassign (ghb, V, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, V, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_subassign (ghb, v, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, v, ui, desc) ; assert (isequal (V1, V2)) ;

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
V2 = gtb (ghb, V) ;
V2 (i) = S ;

s = v (i) ;
t = ui ;
s (wi) = t (wi) ;
v2 = v ;
v2 (i) = s ;
assert (isequal (v2, V2)) ;

V1 = gtb_subassign (ghb, V, Wi, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, Wi, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, I, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, V, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_subassign (ghb, v, wi, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, wi, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, I, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, v, wi, ui, desc) ; assert (isequal (V1, V2)) ;

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
S = gtb (ghb, T) ;
V2 = gtb (ghb, V) ;
V2 (i) = S ;

V1 = gtb_subassign (ghb, V, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, V, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, V, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, I, V, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, accum, V, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, V, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, V, Ui, accum, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_subassign (ghb, v, accum, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, accum, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, I, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, I, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, v, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, v, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, I, v, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, accum, v, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, v, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, v, ui, accum, desc) ; assert (isequal (V1, V2)) ;

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
V2 = gtb (ghb, V) ;
V2 (i) = S ;

s = v (i) ;
t = s + ui ;
s (wi) = t (wi) ;
v2 = v ;
v2 (i) = s ;
assert (isequal (v2, V2)) ;

V1 = gtb_subassign (ghb, V, Wi, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, Wi, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, Wi, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, Wi, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, Wi, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, Wi, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, accum, Wi, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, accum, Wi, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, accum, I, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, I, Wi, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, I, Wi, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, V, I, accum, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, V, Wi, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, V, Wi, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, V, I, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, I, V, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, V, Wi, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, V, Wi, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, V, accum, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, accum, V, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_subassign (ghb, v, wi, accum, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, wi, accum, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, wi, ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, wi, ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, wi, I, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, wi, I, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, accum, wi, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, accum, wi, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, accum, I, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, I, wi, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, I, wi, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, v, I, accum, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, v, wi, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, v, wi, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, v, I, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, accum, I, v, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, v, wi, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, v, wi, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, v, accum, wi, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_subassign (ghb, I, accum, v, wi, ui, desc) ; assert (isequal (V1, V2)) ;

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
S = gtb (ghb, T) ;
C2 = gtb (ghb, C) ;
C2 (i,j) = S ;

t = aij ;
s = t ;
c2 = c ;
c2 (i,j) = s ;
assert (isequal (c2, C2)) ;

C1 = gtb_subassign (ghb, C, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, C, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Aij, J, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_subassign (ghb, c, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, c, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, aij, J, desc) ; assert (isequal (C1, C2)) ;

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
C2 = gtb (ghb, C) ;
C2 (i,j) = S ;

s = c (i,j) ;
t = aij ;
s (mij) = t (mij) ;
c2 = c ;
c2 (i,j) = s ;
assert (isequal (c2, C2)) ;

C1 = gtb_subassign (ghb, C, Mij, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, C, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_subassign (ghb, c, mij, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, c, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;

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
S = gtb (ghb, T) ;
C2 = gtb (ghb, C) ;
C2 (i,j) = S ;

s = c (i,j) ;
t = s + aij ;
s = t ;
c2 = c ;
c2 (i,j) = s ;
assert (isequal (c2, C2)) ;

C1 = gtb_subassign (ghb, C, accum, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, accum, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, accum, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, accum, C, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, C, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, C, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, C, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, C, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, J, C, Aij, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_subassign (ghb, c, accum, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, accum, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, accum, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, accum, c, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, c, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, c, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, c, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, c, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, c, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, c, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, c, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, J, c, aij, desc) ; assert (isequal (C1, C2)) ;

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
C2 = gtb (ghb, C) ;
C2 (i,j) = S ;

s = c (i,j) ;
t = s + aij ;
s (mij) = t (mij) ;
c2 = c ;
c2 (i,j) = s ;
assert (isequal (c2, C2)) ;

C1 = gtb_subassign (ghb, C, Mij, accum, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, accum, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, accum, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, Aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, Aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, Aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, I, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, I, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, I, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, I, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, I, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, Mij, I, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, accum, Mij, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, accum, Mij, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, accum, Mij, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, accum, I, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, accum, I, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, accum, I, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, J, accum, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, J, Mij, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, J, Mij, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, accum, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, accum, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, accum, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Mij, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Mij, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Mij, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Mij, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Mij, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, C, I, Mij, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, Mij, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, Mij, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, Mij, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, I, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, I, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, C, I, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, J, C, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, C, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, C, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, C, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, accum, C, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, C, accum, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, C, Mij, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, C, Mij, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, accum, J, C, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, accum, C, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, accum, C, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, accum, C, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, accum, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, accum, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, accum, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, J, accum, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, J, Mij, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, J, Mij, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Mij, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Mij, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Mij, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Mij, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Mij, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, C, Mij, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_subassign (ghb, c, mij, accum, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, accum, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, accum, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, I, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, I, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, I, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, I, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, I, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, mij, I, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, accum, mij, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, accum, mij, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, accum, mij, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, accum, I, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, accum, I, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, accum, I, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, J, accum, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, J, mij, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, J, mij, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, accum, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, accum, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, accum, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, mij, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, mij, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, mij, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, mij, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, mij, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, c, I, mij, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, c, mij, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, c, mij, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, c, mij, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, c, I, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, c, I, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, c, I, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, J, c, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, c, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, c, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, accum, I, c, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, accum, c, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, c, accum, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, c, mij, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, J, c, mij, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, accum, J, c, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, accum, c, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, accum, c, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, accum, c, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, accum, J, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, accum, mij, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, accum, mij, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, J, accum, mij, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, J, mij, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, J, mij, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, mij, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, mij, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, mij, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, mij, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, mij, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_subassign (ghb, I, c, mij, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest85 (%d): all tests passed\n', ghb) ;

