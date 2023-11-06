function gbtest92
%GBTEST92 test GrB.kronecker
%
% C = GrB.kronecker (op, A, B)
% C = GrB.kronecker (op, A, B, desc)
% C = GrB.kronecker (C, accum, op, A, B, desc)
% C = GrB.kronecker (C, M, op, A, B, desc)
% C = GrB.kronecker (C, M, accum, op, A, B, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (4*7, 3*9, 0.5) ;
M     = GrB.random (4*7, 3*9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GrB.random (4, 3, 0.5) ;
B     = GrB.random (7, 9, 0.5) ;
desc  = struct ;

op = '*' ;

c = double (C) ;
m = logical (M) ;
a = double (A) ;
b = double (B) ;

%----------------------------------------------------------------------
% C = GrB.kronecker (op, A, B)
%----------------------------------------------------------------------

% 2 matrices: A, B
% 1 string: op

C2 = kron (A,B) ;
c2 = kron (a,b) ;
assert (isequal (c2, C2)) ;

C1 = GrB.kronecker (op, A, B) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (A, op, B) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (A, B, op) ; assert (isequal (C1, C2)) ;

C1 = GrB.kronecker (op, a, b) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (a, op, b) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (a, b, op) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.kronecker (op, A, B, desc)
%----------------------------------------------------------------------

% 2 matrices: A, B
% 1 string: op

C2 = kron (A,B) ;
c2 = kron (a,b) ;
assert (isequal (c2, C2)) ;

C1 = GrB.kronecker (op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (A, B, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.kronecker (op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (a, b, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.kronecker (C, accum, op, A, B, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 2 strings: accum, op

% C = accum (C, kron (A,B)) ;

C2 = C + kron (A,B) ;
c2 = c + kron (a,b) ;
assert (isequal (c2, C2)) ;

C1 = GrB.kronecker (C, accum, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, accum, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, accum, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, A, accum, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, A, accum, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, A, B, accum, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.kronecker (c, accum, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, accum, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, accum, a, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, a, accum, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, a, accum, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, a, b, accum, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.kronecker (C, M, op, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 1 string: op

% C<M> = kron (A,B) ;

T = kron (A,B) ;
C2 = C ;
C2 (M) = T (M) ;

t = kron (a,b) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.kronecker (op, C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.kronecker (op, c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, m, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, m, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, m, a, b, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.kronecker (C, M, accum, op, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 2 string: accum, op

% C<M> = accum (C, kron (A,B)) ;

T = C + kron (A,B) ;
C2 = C ;
C2 (M) = T (M) ;

t = c + kron (a,b) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.kronecker (C, M, accum, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, M, accum, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, M, accum, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, accum, op, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, accum, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, accum, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, accum, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, M, A, B, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, M, A, accum, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (C, M, A, accum, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (accum, op, C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (accum, C, op, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (accum, C, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (accum, C, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (accum, C, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.kronecker (c, m, accum, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, m, accum, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, m, accum, a, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, accum, op, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, accum, m, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, accum, m, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, accum, m, a, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, m, a, b, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, m, a, accum, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (c, m, a, accum, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (accum, op, c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (accum, c, op, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (accum, c, m, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (accum, c, m, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.kronecker (accum, c, m, a, b, op, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest92: all tests passed\n') ;

