function gbtest135
%GBTEST135 test GhB.apply2 (inplace usage)
%
% GhB.apply2 (C, op, A, B)                     % C = op(A,B)
% GhB.apply2 (C, accum, op, A, B)              % C += op(A,B)
% GhB.apply2 (C, M, op, A, B)                  % C<M> = op(A,B)
% GhB.apply2 (C, M, accum, op, A, B)           % C<M> += op(A,B)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C     = GhB.random (9, 9, 0.5) ;
M     = GhB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
mult  = '*' ;
div   = '/' ;
A     = GhB.random (9, 9, 0.5) ;
x     = exp (1) ;
y     = pi ;
desc  = struct ;
C0    = GhB (9, 9) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, op, A, y)
%----------------------------------------------------------------------

% 3 matrices: C, A, y
% 1 string: op

C2 = A / y ;
C3 = GhB.apply2 (C, div, A, y)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.apply2 (C1, div, A, y) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.apply2 (C1, A, div, y) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.apply2 (C1, A, y, div) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, op, A, y, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, y
% 1 string: op

C2 = A / y ;
C3 = GhB.apply2 (C, div, A, y)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.apply2 (C1, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.apply2 (C1, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.apply2 (C1, A, y, div, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, op, x, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, x, A
% 1 string: op

C2 = x * A ;
C3 = GhB.apply2 (C, mult, x, A)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.apply2 (C1, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.apply2 (C1, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.apply2 (C1, x, A, mult, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, accum, op, A, y)
%----------------------------------------------------------------------

% 3 matrices: C, A, y
% 2 strings: accum, op

C2 = C + A / y ;
C3 = GhB.apply2 (C, accum, div, A, y)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply2 (C1, accum, div, A, y) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, A, div, y) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, A, y, div) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, A, y, accum, div) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, A, y, div) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, A, div, y) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, div, A, y) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, div, C1, A, y) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, accum, op, A, y, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, y
% 2 strings: accum, op

C2 = C + A / y ;
C3 = GhB.apply2 (C, accum, div, A, y)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply2 (C1, accum, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, A, y, accum, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, div, C1, A, y, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, accum, op, x, A)
%----------------------------------------------------------------------

% 3 matrices: C, x, A
% 2 strings: accum, op

C2 = C + x * A ;
C3 = GhB.apply2 (C, accum, mult, x, A)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply2 (C1, accum, mult, x, A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, x, mult, A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, x, A, mult) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, x, A, accum, mult) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, x, A, mult) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, x, mult, A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, mult, x, A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, mult, C1, x, A) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, accum, op, x, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, x, A
% 2 strings: accum, op

C2 = C + x * A ;
C3 = GhB.apply2 (C, accum, mult, x, A, desc)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply2 (C1, accum, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, x, A, accum, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, mult, C1, x, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, M, op, A, y, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, y
% 1 string:   op

% C<M> = A / y
C2 = GhB.assign (C, M, A / y) ;
C3 = GhB.apply2 (C, M, div, A, y, desc)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply2 (C1, M, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, div, M, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (div, C1, M, A, y, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, M, op, x, A, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, x, A
% 1 string:   op

% C<M> = x * A
C2 = GhB.assign (C, M, x * A) ;
C3 = GhB.apply2 (C, M, mult, x, A, desc)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply2 (C1, M, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, mult, M, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (mult, C1, M, x, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, M, accum, op, A, y, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, y
% 2 strings:  accum, op

% C<M> += A / y
C2 = GhB.assign (C, M, accum, A / y) ;
C3 = GhB.apply2 (C, M, accum, div, A, y, desc)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply2 (C1, M, accum, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, accum, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, accum, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, A, y, accum, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, A, accum, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, A, accum, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, M, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, M, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, M, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, div, M, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, M, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, M, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, M, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, div, M, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, div, C1, M, A, y, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply2 (C, M, accum, op, x, A, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, x, A
% 2 strings:  accum, op

% C<M> += x * A
C2 = GhB.assign (C, M, accum, x * A) ;
C3 = GhB.apply2 (C, M, accum, mult, x, A, desc)  ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply2 (C1, M, accum, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, accum, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, accum, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, x, A, accum, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, x, accum, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, M, x, accum, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, M, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, M, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, M, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (C1, accum, mult, M, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, M, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, M, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, M, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, C1, mult, M, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply2 (accum, mult, C1, M, x, A, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest135: all tests passed\n') ;

