function gbtest119
%GBTEST119 test GrB.eunion
%
% C = GrB.eunion (op, A, alpha, B, beta)
% C = GrB.eunion (op, A, alpha, B, beta, desc)
% C = GrB.eunion (C, accum, op, A, alpha, B, beta, desc)
% C = GrB.eunion (C, M, op, A, alpha, B, beta, desc)
% C = GrB.eunion (C, M, accum, op, A, alpha, B, beta, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (9, 9, 0.5) ;
M     = GrB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GrB.random (9, 9, 0.5) ;
B     = GrB.random (9, 9, 0.5) ;
desc  = struct ;

op = '-' ;
alpha = 0 ;
beta = 0 ;

c = double (C) ;
m = logical (M) ;
a = double (A) ;
b = double (B) ;

%----------------------------------------------------------------------
% C = GrB.eunion (op, A, alpha, B, beta)
%----------------------------------------------------------------------

% 4 matrices: A, alpha, B, beta
% 1 string: op

C2 = A-B ;
c2 = a-b ;
assert (isequal (c2, C2)) ;

C1 = GrB.eunion (op, A, alpha, B, beta) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (A, alpha, op, B, beta) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (A, alpha, B, beta, op) ; assert (isequal (C1, C2)) ;

C1 = GrB.eunion (op, a, alpha, b, beta) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (a, alpha, op, b, beta) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (a, alpha, b, beta, op) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eunion (op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 4 matrices: A, alpha, B, beta
% 1 string: op

C2 = A-B ;
c2 = a-b ;
assert (isequal (c2, C2)) ;

C1 = GrB.eunion (op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.eunion (op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eunion (C, accum, op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 5 matrices: C, A, alpha, B, beta
% 2 strings: accum, op

% C = accum (C, op (A,B)) ;

C2 = C + (A-B) ;
c2 = c + (a-b) ;
assert (isequal (c2, C2)) ;

C1 = GrB.eunion (C, accum, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, accum, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, accum, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, A, alpha, accum, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, A, alpha, accum, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, A, alpha, B, beta, accum, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.eunion (c, accum, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, accum, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, accum, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, a, alpha, accum, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, a, alpha, accum, b, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, a, alpha, b, beta, accum, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eunion (C, M, op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 6 matrices: C, M, A, alpha, B, beta
% 1 string: op

% C<M> = op (A,B)

T = (A-B) ;
C2 = C ;
C2 (M) = T (M) ;

t = (a-b) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.eunion (op, C, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, M, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, M, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, M, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.eunion (op, c, m, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, m, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, m, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, m, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eunion (C, M, accum, op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 6 matrices: C, M, A, alpha, B, beta
% 2 string: accum, op

% C<M> = accum (C, A*B) ;

T = C + (A-B) ;
C2 = C ;
C2 (M) = T (M) ;

t = c + (a-b) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.eunion (C, M, accum, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, M, accum, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, M, accum, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, accum, op, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, accum, M, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, accum, M, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, accum, M, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, M, A, alpha, B, beta, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, M, A, alpha, accum, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (C, M, A, alpha, accum, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (accum, op, C, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (accum, C, op, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (accum, C, M, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (accum, C, M, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (accum, C, M, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.eunion (c, m, accum, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, m, accum, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, m, accum, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, accum, op, m, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, accum, m, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, accum, m, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, accum, m, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, m, a, alpha, b, beta, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, m, a, alpha, accum, b, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (c, m, a, alpha, accum, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (accum, op, c, m, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (accum, c, op, m, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (accum, c, m, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (accum, c, m, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eunion (accum, c, m, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;

beta = GrB (beta) ;
alpha = GrB (alpha) ;
C1 = GrB.eunion (accum, c, m, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest119: all tests passed\n') ;

