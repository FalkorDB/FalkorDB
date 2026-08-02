function gbtest145
%GBTEST145 test GhB.trans (inplace usage)
%
% GhB.trans (C, A)                     C = A'
% GhB.trans (C, accum, A)              C += A'
% GhB.trans (C, M, A)                  C<M> = A'
% GhB.trans (C, M, accum, A)           C<M> += A'

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C      = GhB.random (8, 9, 0.5) ;
M      = GhB.random (8, 9, 0.5, 'range', logical ([false true])) ;
accum  = '+' ;
A      = GhB.random (9, 8, 0.5) ;
desc   = struct ;
C0     = GhB (8, 9) ;

%----------------------------------------------------------------------
% GhB.trans (C, A)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 0 string:

C2 = A.' ;
C3 = GhB.trans (A) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.trans (C1, A) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.trans (C, A, desc)
%----------------------------------------------------------------------

% 2 matrices C, A
% 0 string:

C2 = A.' ;
C3 = GhB.trans (A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C0) ; GhB.trans (C1, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.trans (C, accum, A, desc)
%----------------------------------------------------------------------

% 2 matrices C, A
% 1 string: accum

C2 = C + A.' ;
C3 = GhB.trans (C, accum, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.trans (C1, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.trans (C1, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.trans (accum, C1, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.trans (C, M, A, desc)
%----------------------------------------------------------------------

% 3 matrices C, M, A

% C<M> = A.'

C2 = GhB (C) ;
T = A.' ;
C2 (M) = T (M) ;
C3 = GhB.trans (C, M, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.trans (C1, M, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.trans (C, M, accum, A, desc)
%----------------------------------------------------------------------

% 3 matrices C, M, A
% 1 string: accum

% C<M> += A.'

C2 = GhB (C) ;
T = C + A.' ;
C2 (M) = T (M) ;
C3 = GhB.trans (C, M, accum, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.trans (C1, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.trans (accum, C1, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.trans (C1, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.trans (C1, M, A, accum, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest145: all tests passed\n') ;

