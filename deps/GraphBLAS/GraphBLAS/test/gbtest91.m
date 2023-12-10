function gbtest91
%GBTEST91 test GrB.trans
%
% C = GrB.trans (A)
% C = GrB.trans (A, desc)
% C = GrB.trans (C, accum, A, desc)
% C = GrB.trans (C, M, A, desc)
% C = GrB.trans (C, M, accum, A, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C      = GrB.random (8, 9, 0.5) ;
M      = GrB.random (8, 9, 0.5, 'range', logical ([false true])) ;
accum  = '+' ;
A      = GrB.random (9, 8, 0.5) ;
desc   = struct ;

c = double (C) ;
a = double (A) ;
m = logical (M) ;

%----------------------------------------------------------------------
% C = GrB.trans (A)
%----------------------------------------------------------------------

% 1 matrix: A
% 0 string:

C2 = A.' ;
c2 = a.' ;
assert (isequal (c2, C2)) ;

C1 = GrB.trans (A) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (a) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.trans (A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 0 string:

C2 = A.' ;
c2 = a.' ;
assert (isequal (c2, C2)) ;

C1 = GrB.trans (A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.trans (C, accum, A, desc)
%----------------------------------------------------------------------

% 2 matrices C, A
% 1 string: accum

C2 = C + A.' ;
c2 = c + a.' ;
assert (isequal (c2, C2)) ;

C1 = GrB.trans (C, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (C, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (accum, C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.trans (c, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (c, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (accum, c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.trans (C, M, A, desc)
%----------------------------------------------------------------------

% 3 matrices C, M, A

% C<M> = A.'

C2 = C ;
T = A.' ;
C2 (M) = T (M) ;

c2 = c ;
t = a.' ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.trans (C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.trans (C, M, accum, A, desc)
%----------------------------------------------------------------------

% 3 matrices C, M, A
% 1 string: accum

% C<M> += A.'

C2 = C ;
T = C + A.' ;
C2 (M) = T (M) ;

c2 = c ;
t = c + a.' ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.trans (C, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (accum, C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (C, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (C, M, A, accum, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.trans (c, m, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (accum, c, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (c, accum, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.trans (c, m, a, accum, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest91: all tests passed\n') ;
