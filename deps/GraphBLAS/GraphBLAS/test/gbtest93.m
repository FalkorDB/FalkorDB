function gbtest93 (ghb, ghb2)
%GBTEST93 test [GrB,GhB].select
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

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
if (nargin < 2)
    ghb2 = ghb ;
end
gtb_name = gtb_prep (ghb) ;

C     = gtb_random (ghb2, 9, 9, 0.5, 'range', [-1 1]) ;
M     = gtb_random (ghb2, 9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = gtb_random (ghb2, 9, 9, 0.5, 'range', [-1 1]) ;
B     = gtb (ghb2, 0.5) ;
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

C1 = gtb_select (ghb, '>0', A) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, A, '>0') ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, '>0', a) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, a, '>0') ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (op, A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 1 string: op

C2 = A .* (A > 0) ;
c2 = a .* (a > 0) ;
assert (isequal (c2, C2)) ;

C1 = gtb_select (ghb, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, A, '>0', desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, a, '>0', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (op, A, b, desc)
%----------------------------------------------------------------------

% 2 matrices A, b
% 1 string: op

C2 = A .* (A > 0.5) ;
c2 = a .* (a > 0.5) ;
assert (isequal (c2, C2)) ;

C1 = gtb_select (ghb, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, '>', a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, a, '>', b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, a, b, '>', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, accum, op, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 strings: accum, op

C2 = C + A .* (A > 0) ;
c2 = c + a .* (a > 0) ;
assert (isequal (c2, C2)) ;

C1 = gtb_select (ghb, C, accum, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, accum, A, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, A, accum, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, '>0', C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, c, accum, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, accum, a, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, a, accum, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, '>0', c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, accum, op, A, b, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, b
% 2 strings: accum, op

C2 = C + A .* (A > 0.5) ;
c2 = c + a .* (a > 0.5) ;
assert (isequal (c2, C2)) ;

C1 = gtb_select (ghb, C, accum, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, accum, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, accum, A, B, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, A, B, accum, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, A, accum, B, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, A, accum, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, '>', C, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, C, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, C, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, C, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, c, accum, '>', a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, accum, a, '>', b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, accum, a, b, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, a, b, accum, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, a, accum, b, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, a, accum, '>', b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, '>', c, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, c, '>', a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, c, a, '>', b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, c, a, b, '>', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, M, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string: op

% C<M> = op (A)
T = A .* (A > 0) ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = a .* (a > 0) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_select (ghb, C, M, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, '>0', M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, '>0', C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, M, A, '>0', desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, c, m, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, '>0', m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, '>0', c, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, m, a, '>0', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, M, op, A, b, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, b
% 1 string: op

% C<M> = op (A,b)

T = A .* (A > 0.5) ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = a .* (a > 0.5) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_select (ghb, C, M, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, '>', C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, '>', M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, M, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, M, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, c, m, '>', a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, '>', c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, '>', m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, m, a, '>', b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, m, a, b, '>', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, M, accum, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 strings: accum, op

% C<M> += op (A)

T = C + A .* (A > 0) ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = c + a .* (a > 0) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_select (ghb, C, M, accum, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, accum, '>0', M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, accum, M, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, accum, M, A, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, M, A, accum, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, M, accum, A, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, '>0', C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, C, '>0', M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, C, M, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, C, M, A, '>0', desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, c, m, accum, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, accum, '>0', m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, accum, m, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, accum, m, a, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, m, a, accum, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, c, m, accum, a, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, '>0', c, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, c, '>0', m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, c, m, '>0', a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, c, m, a, '>0', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.select (C, M, accum, op, A, b, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, b
% 2 strings: accum, op

% C<M> += op (A,b)

T = C + A .* (A > 0.5) ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = c + a .* (a > 0.5) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_select (ghb, C, M, accum, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, M, accum, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, M, accum, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, C, M, A, B, accum, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, M, A, accum, B, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, M, A, accum, '>', B, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, C, accum, '>', M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, accum, M, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, accum, M, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, C, accum, M, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_select (ghb, accum, '>', C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, C, '>', M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, C, M, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, C, M, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_select (ghb, accum, C, M, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest93 (%d): all tests passed\n', ghb) ;

