function gbtest97 (ghb, ghb2)
%GBTEST97 test [GrB,GhB].apply2
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

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
if (nargin < 2)
    ghb2 = ghb ;
end
gtb_name = gtb_prep (ghb) ;

C     = gtb_random (ghb2, 9, 9, 0.5) ;
M     = gtb_random (ghb2, 9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
mult  = '*' ;
div   = '/' ;
A     = gtb_random (ghb2, 9, 9, 0.5) ;
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

C1 = gtb_apply2 (ghb, div, A, y) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, A, div, y) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, A, y, div) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, div, a, y) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, a, div, y) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, a, y, div) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (op, A, y, desc)
%----------------------------------------------------------------------

% 2 matrix: A, y
% 1 string: op

C2 = A / y ;

C1 = gtb_apply2 (ghb, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, A, y, div, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, a, y, div, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (op, x, A, desc)
%----------------------------------------------------------------------

% 2 matrix: x, A
% 1 string: op

C2 = x * A ;

C1 = gtb_apply2 (ghb, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, x, A, mult, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, x, a, mult, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, accum, op, A, y)
%----------------------------------------------------------------------

% 3 matrices: C, A, y
% 2 strings: accum, op

C2 = C + A / y ;
c2 = c + a / y ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply2 (ghb, C, accum, div, A, y) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, A, div, y) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, A, y, div) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, A, y, accum, div) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, A, y, div) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, A, div, y) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, div, A, y) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, div, C, A, y) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, c, accum, div, a, y) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, a, div, y) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, a, y, div) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, a, y, accum, div) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, a, y, div) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, accum, op, A, y, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, y
% 2 strings: accum, op

C2 = C + A / y ;
c2 = c + a / y ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply2 (ghb, C, accum, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, A, y, accum, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, div, C, A, y, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, c, accum, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, a, y, accum, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, div, c, a, y, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, accum, op, x, A)
%----------------------------------------------------------------------

% 3 matrices: C, x, A
% 2 strings: accum, op

C2 = C + x * A ;
c2 = c + x * a ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply2 (ghb, C, accum, mult, x, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, x, mult, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, x, A, mult) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, x, A, accum, mult) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, x, A, mult) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, x, mult, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, mult, x, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, mult, C, x, A) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, c, accum, mult, x, a) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, x, mult, a) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, x, a, mult) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, x, a, accum, mult) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, x, a, mult) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, x, mult, a) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, mult, x, a) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, mult, c, x, a) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, accum, op, x, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, x, A
% 2 strings: accum, op

C2 = C + x * A ;
c2 = c + x * a ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply2 (ghb, C, accum, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, x, A, accum, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, mult, C, x, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, c, accum, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, x, a, accum, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, mult, c, x, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, M, op, A, y, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, y
% 1 string:   op

% C<M> = A / y
C2 = gtb_assign (ghb, C, M, A / y) ;

t = a / y ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply2 (ghb, C, M, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, div, M, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, div, C, M, A, y, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, c, m, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, m, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, m, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, div, m, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, div, c, m, a, y, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, M, op, x, A, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, x, A
% 1 string:   op

% C<M> = x * A
C2 = gtb_assign (ghb, C, M, x * A) ;

t = x * a ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply2 (ghb, C, M, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, mult, M, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, mult, C, M, x, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, c, m, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, m, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, m, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, mult, m, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, mult, c, m, x, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, M, accum, op, A, y, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, y
% 2 strings:  accum, op

% C<M> += A / y
C2 = gtb_assign (ghb, C, M, accum, A / y) ;

t = c + a / y ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply2 (ghb, C, M, accum, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, accum, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, accum, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, A, y, accum, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, A, accum, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, A, accum, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, M, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, M, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, M, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, div, M, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, M, div, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, M, A, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, M, A, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, div, M, A, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, div, C, M, A, y, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, c, M, accum, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, M, accum, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, M, accum, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, M, a, y, accum, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, M, a, accum, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, M, a, accum, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, M, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, M, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, M, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, div, M, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, M, div, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, M, a, div, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, M, a, y, div, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, div, M, a, y, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, div, c, M, a, y, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply2 (C, M, accum, op, x, A, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, x, A
% 2 strings:  accum, op

% C<M> += x * A
C2 = gtb_assign (ghb, C, M, accum, x * A) ;

t = c + x * a ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply2 (ghb, C, M, accum, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, accum, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, accum, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, x, A, accum, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, x, accum, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, M, x, accum, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, M, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, M, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, M, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, C, accum, mult, M, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, M, mult, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, M, x, mult, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, M, x, A, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, C, mult, M, x, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, mult, C, M, x, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply2 (ghb, c, M, accum, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, M, accum, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, M, accum, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, M, x, a, accum, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, M, x, accum, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, M, x, accum, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, M, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, M, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, M, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, c, accum, mult, M, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, M, mult, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, M, x, mult, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, M, x, a, mult, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, c, mult, M, x, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply2 (ghb, accum, mult, c, M, x, a, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest97 (%d): all tests passed\n', ghb) ;

