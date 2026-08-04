function gbtest138
%GBTEST138 test GhB.eunion (inplace usage)
%
% GhB.eunion (C, op, A, alpha, B, beta)                  C = op(A,alpha,B,beta)
% GhB.eunion (C, accum, op, A, alpha, B, beta)           C += op(...)
% GhB.eunion (C, M, op, A, alpha, B, beta)               C<M> = op(...)
% GhB.eunion (C, M, accum, op, A, alpha, B, beta)        C<M> += op(...)

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

op = '-' ;
alpha = 0 ;
beta = 0 ;

%----------------------------------------------------------------------
% GhB.eunion (C, op, A, alpha, B, beta)
%----------------------------------------------------------------------

% 4 matrices: A, alpha, B, beta
% 1 string: op

C2 = A-B ;
C3 = GhB.eunion (op, A, alpha, B, beta) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.eunion (C1, op, A, alpha, B, beta) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.eunion (C1, A, alpha, op, B, beta) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.eunion (C1, A, alpha, B, beta, op) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.eunion (op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 4 matrices: A, alpha, B, beta
% 1 string: op

C2 = A-B ;

C1 = GhB (C0) ; GhB.eunion (C1, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.eunion (C1, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C0) ; GhB.eunion (C1, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.eunion (C, accum, op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 5 matrices: C, A, alpha, B, beta
% 2 strings: accum, op

% C = accum (C, op (A,B)) ;

C2 = C + (A-B) ;
C3 = GhB.eunion (C, accum, op, A, alpha, B, beta) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.eunion (C1, accum, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, accum, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, accum, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, A, alpha, accum, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, A, alpha, accum, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, A, alpha, B, beta, accum, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.eunion (C, M, op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 6 matrices: C, M, A, alpha, B, beta
% 1 string: op

% C<M> = op (A,B)

T = (A-B) ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.eunion (C, M, op, A, alpha, B, beta, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.eunion (op, C1, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, M, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, M, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, M, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.eunion (C, M, accum, op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 6 matrices: C, M, A, alpha, B, beta
% 2 string: accum, op

% C<M> = accum (C, A*B) ;

T = C + (A-B) ;
C2 = GhB (C) ;
C2 (M) = T (M) ;
C3 = GhB.eunion (C, M, accum, op, A, alpha, B, beta, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.eunion (C1, M, accum, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, M, accum, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, M, accum, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, accum, op, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, accum, M, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, accum, M, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, accum, M, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, M, A, alpha, B, beta, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, M, A, alpha, accum, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (C1, M, A, alpha, accum, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (accum, op, C1, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (accum, C1, op, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (accum, C1, M, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (accum, C1, M, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.eunion (accum, C1, M, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest138: all tests passed\n') ;

