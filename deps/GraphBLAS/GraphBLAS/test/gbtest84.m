function gbtest84 (ghb)
%GBTEST84 test [GrB,GhB].assign
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

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

C     = gtb_random (ghb, 9, 9, 0.5) ;
M     = gtb_random (ghb, 9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = gtb_random (ghb, 9, 9, 0.5) ;
I     = { [1 4 2 5] } ;
J     = { [3 2 7 ] } ;
desc  = struct ;

Aij   = gtb_random (ghb, 4, 3, 0.5) ;

V     = gtb_random (ghb, 9, 1, 0.7) ;
W     = gtb_random (ghb, 9, 1, 0.7, 'range', logical ([false true])) ;

Ui    = gtb_random (ghb, 4, 1, 0.7) ;

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

C2 = gtb (ghb, A) ;

C1 = gtb_assign (ghb, C, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, a) ; assert (isequal (C1, C2)) ;

C1 = gtb_assign (ghb, C, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, M, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, A
% 0 strings:

C2 = gtb (ghb, C) ;
C2 (M) = A (M) ;

c2 = c ;
c2 (m) = a (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_assign (ghb, C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, accum, A) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string:   accum

% C += A

C2 = C + A ;

c2 = c + a ;
assert (isequal (c2, C2)) ;

C1 = gtb_assign (ghb, C, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, C, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.assign (C, M, accum, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string:   accum

% C<M> += A
C2 = gtb (ghb, C) ;
C2 (M) = C2 (M) + A (M) ;

c2 = c ;
c2 (m) = c2 (m) + a (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_assign (ghb, C, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, M, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, C, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_assign (ghb, c, m, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, m, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, accum, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% V = GrB.assign (V, U, I) ;
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

V1 = gtb_assign (ghb, V, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, V, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_assign (ghb, v, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, v, ui, desc) ; assert (isequal (V1, V2)) ;

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
S = gtb (ghb, Ui) ;
Z = gtb (ghb, V) ;
Z (i) = S ;
% with mask:
V2 = gtb (ghb, V) ;
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

V1 = gtb_assign (ghb, V, W, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, W, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, I, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, V, W, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_assign (ghb, v, w, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, w, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, I, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, v, w, ui, desc) ; assert (isequal (V1, V2)) ;

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
Z = gtb (ghb, V) ;
Z (i) = S ;
% with mask:
% V2 = V ;
% V2 (W) = Z (W) ;
% with no mask:
V2 = gtb (ghb, Z) ;

s = v (i) ;
s = s + ui ;
z = v ;
z (i) = s ;
v2 = z ;
assert (isequal (v2, V2)) ;

V1 = gtb_assign (ghb, V, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, V, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, V, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, I, V, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, accum, V, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, V, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, V, Ui, accum, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_assign (ghb, v, accum, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, accum, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, I, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, I, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, v, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, v, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, I, v, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, accum, v, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, v, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, v, ui, accum, desc) ; assert (isequal (V1, V2)) ;

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
Z = gtb (ghb, V) ;
Z (i) = S ;
% with mask:
V2 = gtb (ghb, V) ;
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

V1 = gtb_assign (ghb, V, W, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, W, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, W, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, W, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, W, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, W, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, accum, W, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, accum, W, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, accum, I, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, I, W, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, I, W, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, V, I, accum, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, V, W, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, V, W, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, V, I, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, I, V, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, V, W, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, V, W, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, V, accum, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, accum, V, W, Ui, desc) ; assert (isequal (V1, V2)) ;

V1 = gtb_assign (ghb, v, w, accum, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, w, accum, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, w, ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, w, ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, w, I, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, w, I, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, accum, w, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, accum, w, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, accum, I, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, I, w, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, I, w, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, v, I, accum, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, v, w, ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, v, w, I, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, v, I, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, accum, I, v, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, v, w, ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, v, w, accum, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, v, accum, w, ui, desc) ; assert (isequal (V1, V2)) ;
V1 = gtb_assign (ghb, I, accum, v, w, ui, desc) ; assert (isequal (V1, V2)) ;

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
S = gtb (ghb, Aij) ;
Z = gtb (ghb, C) ;
Z (i,j) = S ;
% with mask:
% C2 = gtb (ghb, C) ;
% C2 (M) = Z (M) ;
% with no mask:
C2 = gtb (ghb, Z) ;

% s = c (i,j) ;
s = aij ;
z = c ;
z (i,j) = s ;
c2 = z ;
assert (isequal (c2, C2)) ;

C1 = gtb_assign (ghb, C, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, J, C, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, Aij, J, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_assign (ghb, c, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, J, c, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, aij, J, desc) ; assert (isequal (C1, C2)) ;

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
S = gtb (ghb, Aij) ;
Z = gtb (ghb, C) ;
Z (i,j) = S ;
% with mask:
C2 = gtb (ghb, C) ;
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

C1 = gtb_assign (ghb, C, M, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, M, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, M, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, J, C, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_assign (ghb, c, m, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, m, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, m, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, J, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, m, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, m, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, J, c, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, J, m, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, m, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, m, aij, J, desc) ; assert (isequal (C1, C2)) ;

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
Z = gtb (ghb, C) ;
Z (i,j) = S ;
% with mask:
% C2 = gtb (ghb, C) ;
% C2 (M) = Z (M) ;
% with no mask:
C2 = gtb (ghb, Z) ;

s = c (i,j) ;
s = s + aij ;
z = c ;
z (i,j) = s ;
c2 = z ;
assert (isequal (c2, C2)) ;

C1 = gtb_assign (ghb, C, accum, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, accum, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, accum, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, Aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, Aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, Aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, C, I, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, J, accum, C, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, J, C, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, J, C, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, C, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, C, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, C, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, C, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, I, C, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, I, C, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, I, J, C, Aij, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_assign (ghb, c, accum, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, accum, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, accum, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, c, I, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, J, accum, c, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, J, c, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, J, c, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, J, accum, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, J, aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, accum, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, accum, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, I, c, aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, c, aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, c, I, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, c, I, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, I, c, aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, I, c, J, aij, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_assign (ghb, accum, I, J, c, aij, desc) ; assert (isequal (C1, C2)) ;

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
Z = gtb (ghb, C) ;
Z (i,j) = S ;
% with mask:
C2 = gtb (ghb, C) ;
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

C1 = gtb_assign (ghb, C, M, accum, Aij, I, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, accum, I, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, accum, I, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, Aij, accum, I, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, Aij, I, accum, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, Aij, I, J, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, I, J, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, I, J, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, I, Aij, J, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, I, Aij, accum, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, I, accum, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, M, I, accum, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, accum, M, Aij, I, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, accum, M, I, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, accum, M, I, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, accum, I, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, accum, I, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, accum, I, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, J, accum, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, J, M, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, J, M, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, accum, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, accum, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, accum, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, M, accum, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, M, accum, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, M, Aij, accum, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, M, Aij, J, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, M, J, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, C, I, M, J, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, C, M, Aij, I, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, C, M, I, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, C, M, I, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, C, I, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, C, I, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, C, I, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, I, J, C, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, I, C, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, I, C, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, I, C, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, J, accum, C, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, J, C, accum, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, J, C, M, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, J, C, M, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, accum, J, C, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, accum, C, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, accum, C, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, accum, C, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, accum, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, accum, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, accum, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, J, accum, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, J, M, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, J, M, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, M, J, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, M, J, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, M, accum, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, M, accum, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, M, Aij, accum, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, C, M, Aij, J, accum, desc) ; assert (isequal (C1, C2));

C1 = gtb_assign (ghb, c, m, accum, aij, I, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, accum, I, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, accum, I, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, aij, accum, I, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, aij, I, accum, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, aij, I, J, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, I, J, aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, I, J, accum, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, I, aij, J, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, I, aij, accum, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, I, accum, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, m, I, accum, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, accum, m, aij, I, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, accum, m, I, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, accum, m, I, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, accum, I, J, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, accum, I, m, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, accum, I, m, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, J, accum, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, J, m, accum, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, J, m, aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, accum, J, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, accum, m, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, accum, m, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, m, accum, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, m, accum, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, m, aij, accum, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, m, aij, J, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, m, J, aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, c, I, m, J, accum, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, c, m, aij, I, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, c, m, I, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, c, m, I, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, c, I, m, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, c, I, m, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, c, I, J, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, I, J, c, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, I, c, J, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, I, c, m, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, accum, I, c, m, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, J, accum, c, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, J, c, accum, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, J, c, m, accum, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, J, c, m, aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, accum, J, c, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, accum, c, J, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, accum, c, m, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, accum, c, m, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, accum, J, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, accum, m, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, accum, m, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, J, accum, m, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, J, m, accum, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, J, m, aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, m, J, accum, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, m, J, aij, accum, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, m, accum, J, aij, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, m, accum, aij, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, m, aij, accum, J, desc) ; assert (isequal (C1, C2));
C1 = gtb_assign (ghb, I, c, m, aij, J, accum, desc) ; assert (isequal (C1, C2));

fprintf ('gbtest84 (%d): all tests passed\n', ghb) ;

