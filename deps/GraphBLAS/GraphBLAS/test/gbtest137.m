function gbtest137
%GBTEST137 test GhB.subassign (inplace usage)
%
% GhB.subassign (C, A, I, J)                   C(I,J) = A
% GhB.subassign (C, accum, A, I, J)            C(I,J) += A
% GhB.subassign (C, M, A, I, J)                C(I,J)<M> = A
% GhB.subassign (C, M, accum, A, I, J)         C(I,J)<M> += A

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C     = GhB.random (9, 9, 0.5) ;
M     = GhB.random (9, 9, 0.5, 'range', logical ([false true])) ;
Mij   = GhB.random (4, 3, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GhB.random (9, 9, 0.5) ;
I     = { [1 4 2 5] } ;
J     = { [3 2 7 ] } ;
desc  = struct ;
C0    = GhB (9, 9) ;

Aij   = GhB.random (4, 3, 0.5) ;
V     = GhB.random (9, 1, 0.7) ;
Wi    = GhB.random (4, 1, 0.7, 'range', logical ([false true])) ;
Ui    = GhB.random (4, 1, 0.7) ;

i = I {1} ;
j = J {1} ;

%----------------------------------------------------------------------
% GhB.subassign (C, A) ;
%----------------------------------------------------------------------

% 1 matrix: A
% 0 strings:

C2 = GhB (A) ;
C3 = GhB.subassign (C, A) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.subassign (C1, A) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.subassign (C, M, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, A
% 0 strings:

C2 = GhB (C) ;
C2 (M) = A (M) ;
C3 = GhB.subassign (C, M, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.subassign (C1, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, M, A      ) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.subassign (C, accum, A) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string:   accum

% C += A

C2 = C + A ;
C3 = GhB.subassign (C, accum, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.subassign (C1, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.subassign (C, M, accum, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string:   accum

% C<M> += A
C2 = GhB (C) ;
C2 (M) = C2 (M) + A (M) ;
C3 = GhB.subassign (C, M, accum, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.subassign (C1, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, M, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, M, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.subassign (V, U, I) ;
%----------------------------------------------------------------------

% 2 vectors: V, U
% 0 strings:
% 1 index:   I

% V(I) = U

V2 = GhB (V) ;
V2 (i) = Ui ;

V3 = GhB.subassign (V, Ui, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.subassign (V1, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (I, V1, Ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.subassign (V, W, U, I) ;
%----------------------------------------------------------------------

% 3 vectors: V, Wi, Ui
% 0 strings:
% 1 index:   I

% V(I)<W> = Ui

S = V (i) ;
% with accum
% T = S + Ui ;
% with no accum:
T = GhB (Ui) ;
% with mask:
S (Wi) = T (Wi) ;
% with no mask:
% S = T ;
V2 = GhB (V) ;
V2 (i) = S ;

V3 = GhB.subassign (V, Wi, Ui, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.subassign (V1, Wi, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, Wi, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, I, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (I, V1, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.subassign (V, accum, U, I) ;
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
S = GhB (T) ;
V2 = GhB (V) ;
V2 (i) = S ;

V3 = GhB.subassign (V, accum, Ui, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.subassign (V1, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (accum, V1, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (accum, V1, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (accum, I, V1, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (I, accum, V1, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (I, V1, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (I, V1, Ui, accum, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.subassign (V, W, accum, U, I) ;
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
V2 = GhB (V) ;
V2 (i) = S ;

V3 = GhB.subassign (V, Wi, accum, Ui, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.subassign (V1, Wi, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, Wi, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, Wi, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, Wi, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, Wi, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, Wi, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, accum, Wi, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, accum, Wi, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, accum, I, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, I, Wi, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, I, Wi, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (V1, I, accum, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (accum, V1, Wi, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (accum, V1, Wi, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (accum, V1, I, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (accum, I, V1, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (I, V1, Wi, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (I, V1, Wi, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (I, V1, accum, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.subassign (I, accum, V1, Wi, Ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.subassign (C, A, I, J) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 indices:  I, J

% C(I,J)<M> = accum (C(I,J), Aij)

% S = C (i,j) ;
% with accum:
% T = S + Aij ;
% with no accum:
T = GhB (Aij) ;
% with mask:
% S (Mij) = T (Mij) ;
% with no mask:
S = GhB (T) ;
C2 = GhB (C) ;
C2 (i,j) = S ;

C3 = GhB.subassign (C, Aij, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.subassign (C1, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, J, C1, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Aij, J, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.subassign (C, M, A, I, J) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices:  I, J

% C(I,J)<M> = accum (C(I,J), Aij)

S = C (i,j) ;
% with accum:
% T = S + Aij ;
% with no accum:
T = GhB (Aij) ;
% with mask:
S (Mij) = T (Mij) ;
% with no mask:
% S = T ;
C2 = GhB (C) ;
C2 (i,j) = S ;

C3 = GhB.subassign (C, Mij, Aij, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.subassign (C1, Mij, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, J, C1, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.subassign (C, accum, A, I, J) ;
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
S = GhB (T) ;
C2 = GhB (C) ;
C2 (i,j) = S ;

C3 = GhB.subassign (C, accum, Aij, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.subassign (C1, accum, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, accum, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, accum, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, J, accum, C1, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, J, C1, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, J, C1, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, I, C1, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, I, C1, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, I, J, C1, Aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.subassign (C, M, accum, A, I, J) ;
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
C2 = GhB (C) ;
C2 (i,j) = S ;

C3 = GhB.subassign (C, Mij, accum, Aij, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.subassign (C1, Mij, accum, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, accum, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, accum, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, Aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, Aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, Aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, I, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, I, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, I, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, I, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, I, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, Mij, I, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, accum, Mij, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, accum, Mij, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, accum, Mij, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, accum, I, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, accum, I, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, accum, I, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, J, accum, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, J, Mij, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, J, Mij, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, accum, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, accum, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, accum, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Mij, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Mij, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Mij, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Mij, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Mij, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (C1, I, Mij, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, Mij, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, Mij, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, Mij, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, I, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, I, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, C1, I, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, I, J, C1, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, I, C1, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, I, C1, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (accum, I, C1, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, J, accum, C1, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, J, C1, accum, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, J, C1, Mij, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, J, C1, Mij, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, accum, J, C1, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, accum, C1, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, accum, C1, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, accum, C1, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, accum, J, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, accum, Mij, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, accum, Mij, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, J, accum, Mij, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, J, Mij, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, J, Mij, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Mij, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Mij, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Mij, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Mij, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Mij, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.subassign (I, C1, Mij, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest137: all tests passed\n') ;

