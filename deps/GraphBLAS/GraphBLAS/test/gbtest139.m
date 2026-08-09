function gbtest139
%GBTEST139 test GhB.emult (inplace usage
%
% GhB.emult (C, op, A, B)                  C = op(A,B)
% GhB.emult (C, accum, op, A, B)           C += op(A,B)
% GhB.emult (C, M, op, A, B)               C<M> = op(A,B)
% GhB.emult (C, M, accum, op, A, B)        C<M> += op(A,B)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C     = GhB.random (9, 9, 0.5) ;
M     = GhB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = GhB.random (9, 9, 0.5) ;
B     = GhB.random (9, 9, 0.5) ;
desc  = struct ;
C0    = GhB (9, 9) ;

op = '*' ;

%----------------------------------------------------------------------
% GhB.emult (C, op, A, B)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 1 string: op

C2 = A.*B ;
C3 = GhB.emult (op, A, B) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.emult (C1, op, A, B) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.emult (C1, A, op, B) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.emult (C1, A, B, op) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.emult (op, A, B, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 1 string: op

C2 = A.*B ;
C3 = GhB.emult (op, A, B) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.emult (C1, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.emult (C1, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.emult (C1, A, B, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.emult (C, accum, op, A, B, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 2 strings: accum, op

% C = accum (C, op (A,B)) ;

C2 = C + A.*B ;
C3 = GhB.emult (C, accum, op, A, B) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.emult (C1, accum, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, accum, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, accum, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, A, accum, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, A, accum, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, A, B, accum, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.emult (C, M, op, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 1 string: op

% C<M> = op (A,B)

T = A.*B ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.emult (C, M, op, A, B, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.emult (op, C1, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.emult (C, M, accum, op, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 2 string: accum, op

% C<M> = accum (C, A*B) ;

T = C + A.*B ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.emult (C, M, accum, op, A, B, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.emult (C1, M, accum, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, M, accum, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, M, accum, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, accum, op, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, accum, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, accum, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, accum, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, M, A, B, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, M, A, accum, B, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (C1, M, A, accum, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (accum, op, C1, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (accum, C1, op, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (accum, C1, M, op, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (accum, C1, M, A, op, B, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.emult (accum, C1, M, A, B, op, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest139: all tests passed\n') ;

