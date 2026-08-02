function gbtest136
%GBTEST136 test GhB.assign (inplace usage)
%
% GhB.assign (C, A, I, J)                      C(I,J) = A
% GhB.assign (C, accum, A, I, J)               C(I,J) += A
% GhB.assign (C, M, A, I, J)                   C<M>(I,J) = A
% GhB.assign (C, M, accum, A, I, J)            C<M>(I,J) += A

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C     = GhB.random (9, 9, 0.5) ;
M     = GhB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GhB.random (9, 9, 0.5) ;
I     = { [1 4 2 5] } ;
J     = { [3 2 7 ] } ;
desc  = struct ;
C0    = GhB (9, 9) ;

Aij   = GhB.random (4, 3, 0.5) ;

V     = GhB.random (9, 1, 0.7) ;
W     = GhB.random (9, 1, 0.7, 'range', logical ([false true])) ;

Ui    = GhB.random (4, 1, 0.7) ;

i = I {1} ;
j = J {1} ;

%----------------------------------------------------------------------
% GhB.assign (C, A) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 0 strings:

C2 = GhB (A) ;

C1 = GhB (C0) ; GhB.assign (C1, A) ; assert (isequal (C1, C2)) ;

C1 = GhB (C0) ; GhB.assign (C1, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.assign (C, M, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 0 strings:

C2 = GhB (C) ;
C2 (M) = A (M) ;
C3 = GhB.assign (C, M, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.assign (C1, M, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.assign (C, accum, A) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string:   accum

% C += A

C2 = C + A ;
C3 = GhB.assign (C, accum, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.assign (C1, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (accum, C1, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.assign (C, M, accum, A) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string:   accum

% C<M> += A
C2 = GhB (C) ;
C2 (M) = C2 (M) + A (M) ;
C3 = GhB.assign (C, M, accum, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.assign (C1, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, M, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (accum, C1, M, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.assign (V, U, I) ;
%----------------------------------------------------------------------

% 2 vectors: V, U
% 0 strings:
% 1 index:   I

% V(I) = U

V2 = GhB (V) ;
V2 (i) = Ui ;
V3 = GhB.assign (V, Ui, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.assign (V1, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (I, V1, Ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.assign (V, W, U, I) ;
%----------------------------------------------------------------------

% 3 vectors: V, W, Ui
% 0 strings:
% 1 index:   I

% V<W>(I) = Ui

% S = V (i) ;
% with accum
% S = S + Ui ;
% with no accum:
S = GhB (Ui) ;
Z = GhB (V) ;
Z (i) = S ;
% with mask:
V2 = GhB (V) ;
V2 (W) = Z (W) ;
% with no mask:
% V2 = Z ;

V3 = GhB.assign (V, W, Ui, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.assign (V1, W, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, W, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, I, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (I, V1, W, Ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.assign (V, accum, U, I) ;
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
Z = GhB (V) ;
Z (i) = S ;
% with mask:
% V2 = V ;
% V2 (W) = Z (W) ;
% with no mask:
V2 = GhB (Z) ;

V3 = GhB.assign (V, accum, Ui, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.assign (V1, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (accum, V1, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (accum, V1, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (accum, I, V1, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (I, accum, V1, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (I, V1, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (I, V1, Ui, accum, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.assign (V, W, accum, U, I) ;
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
Z = GhB (V) ;
Z (i) = S ;
% with mask:
V2 = GhB (V) ;
V2 (W) = Z (W) ;
% with no mask:
% V2 = Z ;

V3 = GhB.assign (V, W, accum, Ui, I, desc) ;
assert (isequal (V2, V3)) ;

V1 = GhB (V) ; GhB.assign (V1, W, accum, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, W, accum, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, W, Ui, accum, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, W, Ui, I, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, W, I, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, W, I, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, accum, W, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, accum, W, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, accum, I, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, I, W, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, I, W, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (V1, I, accum, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (accum, V1, W, Ui, I, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (accum, V1, W, I, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (accum, V1, I, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (accum, I, V1, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (I, V1, W, Ui, accum, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (I, V1, W, accum, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (I, V1, accum, W, Ui, desc) ; assert (isequal (V1, V2)) ;
V1 = GhB (V) ; GhB.assign (I, accum, V1, W, Ui, desc) ; assert (isequal (V1, V2)) ;

%----------------------------------------------------------------------
% GhB.assign (C, A, I, J) ;
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 indices:  I, J

% C<M>(I,J) = accum (C(I,J), Aij)

% S = C (i,j) ;
% with accum:
% S = S + Aij ;
% with no accum:
S = GhB (Aij) ;
Z = GhB (C) ;
Z (i,j) = S ;
% with mask:
% C2 = GhB (C) ;
% C2 (M) = Z (M) ;
% with no mask:
C2 = GhB (Z) ;

C3 = GhB.assign (C, Aij, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.assign (C1, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, J, C1, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, Aij, J, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.assign (C, M, A, I, J) ;
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 indices:  I, J

% C<M>(I,J) = accum (C(I,J), Aij)

% S = C (i,j) ;
% with accum:
% S = S + Aij ;
% with no accum:
S = GhB (Aij) ;
Z = GhB (C) ;
Z (i,j) = S ;
% with mask:
C2 = GhB (C) ;
C2 (M) = Z (M) ;
% with no mask:
% C2 = Z ;

C1 = GhB (C) ; GhB.assign (C1, M, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, M, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, M, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, J, C1, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, J, M, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, M, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, M, Aij, J, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.assign (C, accum, A, I, J) ;
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
Z = GhB (C) ;
Z (i,j) = S ;
% with mask:
% C2 = GhB (C) ;
% C2 (M) = Z (M) ;
% with no mask:
C2 = GhB (Z) ;

C3 = GhB.assign (C, accum, Aij, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.assign (C1, accum, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, accum, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, accum, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, Aij, accum, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, Aij, I, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, Aij, I, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (C1, I, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, J, accum, C1, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, J, C1, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, J, C1, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, J, accum, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, J, Aij, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, accum, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, accum, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, Aij, accum, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (I, C1, Aij, J, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (accum, C1, Aij, I, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (accum, C1, I, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (accum, C1, I, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (accum, I, C1, Aij, J, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (accum, I, C1, J, Aij, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.assign (accum, I, J, C1, Aij, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.assign (C, M, accum, A, I, J) ;
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
Z = GhB (C) ;
Z (i,j) = S ;
% with mask:
C2 = GhB (C) ;
C2 (M) = Z (M) ;
% with no mask:
% C2 = Z ;

C3 = GhB.assign (C, M, accum, Aij, I, J, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.assign (C1, M, accum, Aij, I, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, accum, I, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, accum, I, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, Aij, accum, I, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, Aij, I, accum, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, Aij, I, J, accum, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, I, J, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, I, J, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, I, Aij, J, accum, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, I, Aij, accum, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, I, accum, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, M, I, accum, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, accum, M, Aij, I, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, accum, M, I, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, accum, M, I, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, accum, I, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, accum, I, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, accum, I, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, J, accum, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, J, M, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, J, M, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, accum, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, accum, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, accum, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, M, accum, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, M, accum, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, M, Aij, accum, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, M, Aij, J, accum, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, M, J, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (C1, I, M, J, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (accum, C1, M, Aij, I, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (accum, C1, M, I, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (accum, C1, M, I, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (accum, C1, I, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (accum, C1, I, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (accum, C1, I, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (accum, I, J, C1, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (accum, I, C1, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (accum, I, C1, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (accum, I, C1, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, J, accum, C1, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, J, C1, accum, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, J, C1, M, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, J, C1, M, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, accum, J, C1, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, accum, C1, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, accum, C1, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, accum, C1, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, accum, J, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, accum, M, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, accum, M, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, J, accum, M, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, J, M, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, J, M, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, M, J, accum, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, M, J, Aij, accum, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, M, accum, J, Aij, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, M, accum, Aij, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, M, Aij, accum, J, desc) ; assert (isequal (C1, C2));
C1 = GhB (C) ; GhB.assign (I, C1, M, Aij, J, accum, desc) ; assert (isequal (C1, C2));

fprintf ('gbtest136: all tests passed\n') ;

