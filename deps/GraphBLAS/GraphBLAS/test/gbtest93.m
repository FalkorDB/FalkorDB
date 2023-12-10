function gbtest93
%GBTEST93 test GrB.select
%
% C = GrB.select (op, A)
% C = GrB.select (op, A, b)
% C = GrB.select (op, A, b, desc)
%
% C = GrB.select (C, accum, op, A)
% C = GrB.select (C, accum, op, A, b)
% C = GrB.select (C, accum, op, A, b, desc)
%
% C = GrB.select (C, M, op, A)
% C = GrB.select (C, M, op, A, b)
% C = GrB.select (C, M, op, A, b, desc)
%
% C = GrB.select (C, M, accum, op, A)
% C = GrB.select (C, M, accum, op, A, b)
% C = GrB.select (C, M, accum, op, A, b, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (9, 9, 0.5, 'range', [-1 1]) ;
M     = GrB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GrB.random (9, 9, 0.5, 'range', [-1 1]) ;
B     = GrB (0.5) ;
desc  = struct ;

c = double (C) ;
m = logical (M) ;
a = double (A) ;
b = double (B) ;

%----------------------------------------------------------------------
% C = GrB.select (op, A)
%----------------------------------------------------------------------

% 1 matrix: A
% 1 string: op

C2 = A .* (A > 0) ;
c2 = a .* (a > 0) ;
assert (isequal (c2, C2)) ;

C1 = GrB.select ('>0', A) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (A, '>0') ; assert (isequal (C1, C2)) ;

C1 = GrB.select ('>0', a) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (a, '>0') ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (op, A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 1 string: op

C2 = A .* (A > 0) ;
c2 = a .* (a > 0) ;
assert (isequal (c2, C2)) ;

C1 = GrB.select ('>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (A, '>0', desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.select ('>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (a, '>0', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (op, A, b, desc)
%----------------------------------------------------------------------

% 2 matrices A, b
% 1 string: op

C2 = A .* (A > 0.5) ;
c2 = a .* (a > 0.5) ;
assert (isequal (c2, C2)) ;

C1 = GrB.select ('>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.select ('>', a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (a, '>', b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (a, b, '>', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, accum, op, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 strings: accum, op

C2 = C + A .* (A > 0) ;
c2 = c + a .* (a > 0) ;
assert (isequal (c2, C2)) ;

C1 = GrB.select (C, accum, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, accum, A, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, A, accum, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, '>0', C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.select (c, accum, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, accum, a, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, a, accum, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, '>0', c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, accum, op, A, b, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, b
% 2 strings: accum, op

C2 = C + A .* (A > 0.5) ;
c2 = c + a .* (a > 0.5) ;
assert (isequal (c2, C2)) ;

C1 = GrB.select (C, accum, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, accum, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, accum, A, B, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, A, B, accum, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, A, accum, B, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, A, accum, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, '>', C, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, C, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, C, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, C, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.select (c, accum, '>', a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, accum, a, '>', b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, accum, a, b, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, a, b, accum, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, a, accum, b, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, a, accum, '>', b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, '>', c, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, c, '>', a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, c, a, '>', b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, c, a, b, '>', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, M, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string: op

% C<M> = op (A)
T = A .* (A > 0) ;
C2 = C ;
C2 (M) = T (M) ;

t = a .* (a > 0) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.select (C, M, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, '>0', M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select ('>0', C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, M, A, '>0', desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.select (c, m, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, '>0', m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select ('>0', c, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, m, a, '>0', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, M, op, A, b, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, b
% 1 string: op

% C<M> = op (A,b)

T = A .* (A > 0.5) ;
C2 = C ;
C2 (M) = T (M) ;

t = a .* (a > 0.5) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.select (C, M, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select ('>', C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, '>', M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, M, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, M, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.select (c, m, '>', a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select ('>', c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, '>', m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, m, a, '>', b, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, m, a, b, '>', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, M, accum, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 strings: accum, op

% C<M> += op (A)

T = C + A .* (A > 0) ;
C2 = C ;
C2 (M) = T (M) ;

t = c + a .* (a > 0) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.select (C, M, accum, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, accum, '>0', M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, accum, M, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, accum, M, A, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, M, A, accum, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, M, accum, A, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, '>0', C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, C, '>0', M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, C, M, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, C, M, A, '>0', desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.select (c, m, accum, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, accum, '>0', m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, accum, m, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, accum, m, a, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, m, a, accum, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (c, m, accum, a, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, '>0', c, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, c, '>0', m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, c, m, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, c, m, a, '>0', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, M, accum, op, A, b, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, b
% 2 strings: accum, op

% C<M> += op (A,b)

T = C + A .* (A > 0.5) ;
C2 = C ;
C2 (M) = T (M) ;

t = c + a .* (a > 0.5) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.select (C, M, accum, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, M, accum, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, M, accum, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.select (C, M, A, B, accum, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, M, A, accum, B, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, M, A, accum, '>', B, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.select (C, accum, '>', M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, accum, M, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, accum, M, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (C, accum, M, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.select (accum, '>', C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, C, '>', M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, C, M, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, C, M, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.select (accum, C, M, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest93: all tests passed\n') ;

