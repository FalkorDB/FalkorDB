function gbtest97
%GBTEST97 test GrB.apply2
%
% C = GrB.apply2 (op, A, y)
% C = GrB.apply2 (C, accum, op, A, y)
% C = GrB.apply2 (C, M, op, A, y)
% C = GrB.apply2 (C, M, accum, op, A, y)
%
% C = GrB.apply2 (op, x, A)
% C = GrB.apply2 (C, accum, op, x, A)
% C = GrB.apply2 (C, M, op, x, A)
% C = GrB.apply2 (C, M, accum, op, x, A)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (9, 9, 0.5) ;
M     = GrB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
mult  = '*' ;
div   = '/' ;
A     = GrB.random (9, 9, 0.5) ;
x     = exp (1) ;
y     = pi ;
desc  = struct ;

c = double (C) ;
a = double (A) ;
m = logical (M) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (op, A, y)
%----------------------------------------------------------------------

% 2 matrix: A, y
% 1 string: op

C2 = A / y ;

C1 = GrB.apply2 (div, A, y) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (A, div, y) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (A, y, div) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (div, a, y) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (a, div, y) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (a, y, div) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (op, A, y, desc)
%----------------------------------------------------------------------

% 2 matrix: A, y
% 1 string: op

C2 = A / y ;

C1 = GrB.apply2 (div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (A, y, div, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (a, y, div, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (op, x, A, desc)
%----------------------------------------------------------------------

% 2 matrix: x, A
% 1 string: op

C2 = x * A ;

C1 = GrB.apply2 (mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (x, A, mult, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (x, a, mult, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, accum, op, A, y)
%----------------------------------------------------------------------

% 3 matrices: C, A, y
% 2 strings: accum, op

C2 = C + A / y ;
c2 = c + a / y ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply2 (C, accum, div, A, y) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, A, div, y) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, A, y, div) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, A, y, accum, div) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, A, y, div) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, A, div, y) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, div, A, y) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, div, C, A, y) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (c, accum, div, a, y) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, a, div, y) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, a, y, div) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, a, y, accum, div) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, a, y, div) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, accum, op, A, y, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, y
% 2 strings: accum, op

C2 = C + A / y ;
c2 = c + a / y ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply2 (C, accum, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, A, y, accum, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, div, C, A, y, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (c, accum, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, a, y, accum, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, div, c, a, y, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, accum, op, x, A)
%----------------------------------------------------------------------

% 3 matrices: C, x, A
% 2 strings: accum, op

C2 = C + x * A ;
c2 = c + x * a ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply2 (C, accum, mult, x, A) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, x, mult, A) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, x, A, mult) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, x, A, accum, mult) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, x, A, mult) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, x, mult, A) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, mult, x, A) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, mult, C, x, A) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (c, accum, mult, x, a) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, x, mult, a) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, x, a, mult) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, x, a, accum, mult) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, x, a, mult) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, x, mult, a) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, mult, x, a) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, mult, c, x, a) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, accum, op, x, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, x, A
% 2 strings: accum, op

C2 = C + x * A ;
c2 = c + x * a ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply2 (C, accum, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, x, A, accum, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, mult, C, x, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (c, accum, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, x, a, accum, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, mult, c, x, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, M, op, A, y, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, y
% 1 string:   op

% C<M> = A / y
C2 = GrB.assign (C, M, A / y) ;

t = a / y ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply2 (C, M, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, div, M, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (div, C, M, A, y, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (c, m, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, m, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, m, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, div, m, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (div, c, m, a, y, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, M, op, x, A, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, x, A
% 1 string:   op

% C<M> = x * A
C2 = GrB.assign (C, M, x * A) ;

t = x * a ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply2 (C, M, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, mult, M, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (mult, C, M, x, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (c, m, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, m, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, m, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, mult, m, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (mult, c, m, x, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, M, accum, op, A, y, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, y
% 2 strings:  accum, op

% C<M> += A / y
C2 = GrB.assign (C, M, accum, A / y) ;

t = c + a / y ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply2 (C, M, accum, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, accum, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, accum, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, A, y, accum, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, A, accum, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, A, accum, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, M, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, M, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, M, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, div, M, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, M, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, M, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, M, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, div, M, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, div, C, M, A, y, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (c, M, accum, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, M, accum, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, M, accum, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, M, a, y, accum, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, M, a, accum, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, M, a, accum, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, M, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, M, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, M, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, div, M, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, M, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, M, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, M, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, div, M, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, div, c, M, a, y, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, M, accum, op, x, A, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, x, A
% 2 strings:  accum, op

% C<M> += x * A
C2 = GrB.assign (C, M, accum, x * A) ;

t = c + x * a ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply2 (C, M, accum, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, accum, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, accum, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, x, A, accum, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, x, accum, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, M, x, accum, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, M, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, M, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, M, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (C, accum, mult, M, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, M, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, M, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, M, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, C, mult, M, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, mult, C, M, x, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply2 (c, M, accum, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, M, accum, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, M, accum, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, M, x, a, accum, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, M, x, accum, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, M, x, accum, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, M, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, M, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, M, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (c, accum, mult, M, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, M, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, M, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, M, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, c, mult, M, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply2 (accum, mult, c, M, x, a, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest97: all tests passed\n') ;

