function gbtest88
%GBTEST88 test GrB.emult
%
% C = GrB.emult (op, A, B)
% C = GrB.emult (op, A, B, desc)
% C = GrB.emult (C, accum, op, A, B, desc)
% C = GrB.emult (C, M, op, A, B, desc)
% C = GrB.emult (C, M, accum, op, A, B, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (9, 9, 0.5) ;
M     = GrB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GrB.random (9, 9, 0.5) ;
B     = GrB.random (9, 9, 0.5) ;
desc  = struct ;

op = '*' ;

c = double (C) ;
m = logical (M) ;
a = double (A) ;
b = double (B) ;

%----------------------------------------------------------------------
% C = GrB.emult (op, A, B)
%----------------------------------------------------------------------

% 2 matrices: A, B
% 1 string: op

C2 = A.*B ;
c2 = a.*b ;
assert (isequal (c2, C2)) ;

C1 = GrB.emult (op, A, B) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (A, op, B) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (A, B, op) ; assert (isequal (C1, C2)) ;

C1 = GrB.emult (op, a, b) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (a, op, b) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (a, b, op) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.emult (op, A, B, desc)
%----------------------------------------------------------------------

% 2 matrices: A, B
% 1 string: op

C2 = A.*B ;
c2 = a.*b ;
assert (isequal (c2, C2)) ;

C1 = GrB.emult (op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (A, B, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.emult (op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (a, b, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.emult (C, accum, op, A, B, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 2 strings: accum, op

% C = accum (C, op (A,B)) ;

C2 = C + A.*B ;
c2 = c + a.*b ;
assert (isequal (c2, C2)) ;

C1 = GrB.emult (C, accum, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, accum, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, accum, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, A, accum, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, A, accum, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, A, B, accum, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.emult (c, accum, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, accum, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, accum, a, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, a, accum, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, a, accum, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, a, b, accum, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.emult (C, M, op, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 1 string: op

% C<M> = op (A,B)

T = A.*B ;
C2 = C ;
C2 (M) = T (M) ;

t = a.*b ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.emult (op, C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.emult (op, c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, m, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, m, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, m, a, b, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.emult (C, M, accum, op, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 2 string: accum, op

% C<M> = accum (C, A*B) ;

T = C + A.*B ;
C2 = C ;
C2 (M) = T (M) ;

t = c + a.*b ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.emult (C, M, accum, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, M, accum, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, M, accum, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, accum, op, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, accum, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, accum, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, accum, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, M, A, B, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, M, A, accum, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (C, M, A, accum, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (accum, op, C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (accum, C, op, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (accum, C, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (accum, C, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (accum, C, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.emult (c, m, accum, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, m, accum, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, m, accum, a, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, accum, op, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, accum, m, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, accum, m, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, accum, m, a, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, m, a, b, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, m, a, accum, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (c, m, a, accum, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (accum, op, c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (accum, c, op, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (accum, c, m, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (accum, c, m, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.emult (accum, c, m, a, b, op, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest88: all tests passed\n') ;

