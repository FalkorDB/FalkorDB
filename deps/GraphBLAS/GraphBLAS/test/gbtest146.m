function gbtest146
%GBTEST146 test GhB.select (inplace usage)
%
% GhB.select (C, op, A)                        C = op(A)
% GhB.select (C, accum, op, A)                 C += op(A)
% GhB.select (C, M, op, A)                     C<M> = op(A)
% GhB.select (C, M, accum, op, A)              C<M> += op(A)
%
% GhB.select (C, op, A, b)                     C = op(A,b)
% GhB.select (C, accum, op, A, b)              C += op(A,b)
% GhB.select (C, M, op, A, b)                  C<M> = op(A,b)
% GhB.select (C, M, accum, op, A, b)           C<M> += op(A,b)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C     = GhB.random (9, 9, 0.5, 'range', [-1 1]) ;
M     = GhB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GhB.random (9, 9, 0.5, 'range', [-1 1]) ;
B     = GhB (0.5) ;
C0    = GhB (9, 9) ;
desc  = struct ;

%----------------------------------------------------------------------
% GhB.select (C, op, A)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string: op

C2 = A .* (A > 0) ;
C3 = GhB.select ('>0', A) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.select (C1, '>0', A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.select (C1, A, '>0') ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.select (C, op, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string: op

C2 = A .* (A > 0) ;
C3 = GhB.select ('>0', A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.select (C1, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.select (C1, A, '>0', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.select (C, op, A, b, desc)
%----------------------------------------------------------------------

% 3 matrices C, A, b
% 1 string: op

C2 = A .* (A > 0.5) ;
C3 = GhB.select ('>', A, B, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.select (C1, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.select (C1, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.select (C1, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.select (C, accum, op, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 strings: accum, op

C2 = C + A .* (A > 0) ;
C3 = GhB.select (C, accum, '>0', A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.select (C1, accum, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, accum, A, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, A, accum, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, '>0', C1, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.select (C, accum, op, A, b, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, b
% 2 strings: accum, op

C2 = C + A .* (A > 0.5) ;
C3 = GhB.select (C, accum, '>', A, B, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.select (C1, accum, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, accum, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, accum, A, B, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, A, B, accum, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, A, accum, B, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, A, accum, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, '>', C1, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, C1, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, C1, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, C1, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.select (C, M, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string: op

% C<M> = op (A)
T = A .* (A > 0) ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.select (C, M, '>0', A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.select (C1, M, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, '>0', M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select ('>0', C1, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, M, A, '>0', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.select (C, M, op, A, b, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, b
% 1 string: op

% C<M> = op (A,b)

T = A .* (A > 0.5) ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.select (C, M, '>', A, B, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.select (C1, M, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select ('>', C1, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, '>', M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, M, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, M, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.select (C, M, accum, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 strings: accum, op

% C<M> += op (A)

T = C + A .* (A > 0) ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.select (C, M, accum, '>0', A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.select (C1, M, accum, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, accum, '>0', M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, accum, M, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, accum, M, A, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, M, A, accum, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, M, accum, A, '>0', desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, '>0', C1, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, C1, '>0', M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, C1, M, '>0', A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, C1, M, A, '>0', desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.select (C, M, accum, op, A, b, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, b
% 2 strings: accum, op

% C<M> += op (A,b)

T = C + A .* (A > 0.5) ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.select (C, M, accum, '>', A, B, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.select (C1, M, accum, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, M, accum, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, M, accum, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = GhB (C) ; GhB.select (C1, M, A, B, accum, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, M, A, accum, B, '>', desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, M, A, accum, '>', B, desc) ; assert (isequal (C1, C2)) ;

C1 = GhB (C) ; GhB.select (C1, accum, '>', M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, accum, M, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, accum, M, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (C1, accum, M, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

C1 = GhB (C) ; GhB.select (accum, '>', C1, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, C1, '>', M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, C1, M, '>', A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, C1, M, A, '>', B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.select (accum, C1, M, A, B, '>', desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest146: all tests passed\n') ;

