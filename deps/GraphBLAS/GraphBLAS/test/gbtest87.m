function gbtest87
%GBTEST87 test GrB.eadd
%
% C = GrB.eadd (op, A, B)
% C = GrB.eadd (op, A, B, desc)
% C = GrB.eadd (C, accum, op, A, B, desc)
% C = GrB.eadd (C, M, op, A, B, desc)
% C = GrB.eadd (C, M, accum, op, A, B, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (9, 9, 0.5) ;
M     = GrB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GrB.random (9, 9, 0.5) ;
B     = GrB.random (9, 9, 0.5) ;
desc  = struct ;

op = 'max' ;

c = double (C) ;
m = logical (M) ;
a = double (A) ;
b = double (B) ;

%----------------------------------------------------------------------
% C = GrB.eadd (op, A, B)
%----------------------------------------------------------------------

% 2 matrices: A, B
% 1 string: op

C2 = max (A,B) ;
c2 = max (a,b) ;
assert (isequal (c2, C2)) ;

C1 = GrB.eadd (op, A, B) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (A, op, B) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (A, B, op) ; assert (isequal (C1, C2)) ;

C1 = GrB.eadd (op, a, b) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (a, op, b) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (a, b, op) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eadd (op, A, B, desc)
%----------------------------------------------------------------------

% 2 matrices: A, B
% 1 string: op

C2 = max (A,B) ;
c2 = max (a,b) ;
assert (isequal (c2, C2)) ;

C1 = GrB.eadd (op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (A, B, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.eadd (op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (a, b, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eadd (C, accum, op, A, B, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 2 strings: accum, op

% C = accum (C, op (A,B)) ;

C2 = C + max (A,B) ;
c2 = c + max (a,b) ;
assert (isequal (c2, C2)) ;

C1 = GrB.eadd (C, accum, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, accum, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, accum, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, A, accum, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, A, accum, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, A, B, accum, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.eadd (c, accum, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, accum, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, accum, a, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, a, accum, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, a, accum, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, a, b, accum, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eadd (C, M, op, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 1 string: op

% C<M> = op (A,B)

T = max (A,B) ;
C2 = C ;
C2 (M) = T (M) ;

t = max (a,b) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.eadd (op, C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.eadd (op, c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, m, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, m, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, m, a, b, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eadd (C, M, accum, op, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 2 string: accum, op

% C<M> = accum (C, A*B) ;

T = C + max (A,B) ;
C2 = C ;
C2 (M) = T (M) ;

t = c + max (a,b) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.eadd (C, M, accum, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, M, accum, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, M, accum, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, accum, op, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, accum, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, accum, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, accum, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, M, A, B, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, M, A, accum, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (C, M, A, accum, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (accum, op, C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (accum, C, op, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (accum, C, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (accum, C, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (accum, C, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.eadd (c, m, accum, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, m, accum, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, m, accum, a, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, accum, op, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, accum, m, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, accum, m, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, accum, m, a, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, m, a, b, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, m, a, accum, b, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (c, m, a, accum, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (accum, op, c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (accum, c, op, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (accum, c, m, op, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (accum, c, m, a, op, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.eadd (accum, c, m, a, b, op, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest87: all tests passed\n') ;

