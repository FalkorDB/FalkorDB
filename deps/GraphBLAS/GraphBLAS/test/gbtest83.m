function gbtest83
%GBTEST83 test GrB.apply
%
% C = GrB.apply (op, A)
% C = GrB.apply (C, accum, op, A)
% C = GrB.apply (C, M, op, A)
% C = GrB.apply (C, M, accum, op, A)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

C     = GrB.random (9, 9, 0.5) ;
M     = GrB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
op    = 'sqrt' ;
A     = GrB.random (9, 9, 0.5) ;
desc  = struct ;

c = double (C) ;
a = double (A) ;
m = logical (M) ;

%----------------------------------------------------------------------
% C = GrB.apply (op, A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 1 string: op

C2 = sqrt (A) ;

C1 = GrB.apply (op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (A, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply (op, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (a, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply (C, accum, op, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 strings: accum, op

C2 = C + sqrt (A) ;
c2 = c + sqrt (a) ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply (C, accum, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (C, accum, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (C, A, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (accum, C, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (accum, C, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (accum, op, C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply (c, accum, op, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (c, accum, a, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (c, a, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (accum, c, a, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (accum, c, op, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (accum, op, c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply (C, M, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string:   op

% C<M> = sqrt (A)
C2 = GrB.assign (C, M, sqrt (A)) ;

t = sqrt (a) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply (C, M, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (C, M, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (C, op, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (op, C, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply (c, m, op, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (c, m, a, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (c, op, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (op, c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply (C, M, accum, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 strings:  accum, op

% C<M> += sqrt (A)
C2 = GrB.assign (C, M, accum, sqrt (A)) ;

t = c + sqrt (a) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = GrB.apply (C, M, accum, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (C, M, accum, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (C, M, A, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (C, accum, M, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (C, accum, M, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (C, accum, op, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GrB.apply (accum, C, M, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (accum, C, M, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (accum, C, op, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GrB.apply (accum, op, C, M, A, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest83: all tests passed\n') ;

