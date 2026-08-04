function gbtest86 (ghb)
%GBTEST86 test [GrB,GhB].mxm
%
% C = GrB.mxm (semiring, A, B)
% C = GrB.mxm (semiring, A, B, desc)
% C = GrB.mxm (C, accum, semiring, A, B, desc)
% C = GrB.mxm (C, M, semiring, A, B, desc)
% C = GrB.mxm (C, M, accum, semiring, A, B, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

C     = gtb_random (ghb, 9, 9, 0.5) ;
M     = gtb_random (ghb, 9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
A     = gtb_random (ghb, 9, 9, 0.5) ;
B     = gtb_random (ghb, 9, 9, 0.5) ;
desc  = struct ;

semiring = '+.*' ;

c = double (C) ;
m = logical (M) ;
a = double (A) ;
b = double (B) ;

%----------------------------------------------------------------------
% C = GrB.mxm (semiring, A, B)
%----------------------------------------------------------------------

% 2 matrices: A, B
% 1 string: semiring

C2 = A*B ;
c2 = a*b ;
assert (gbtest_err (c2, C2) < 1e-14) ;

C1 = gtb_mxm (ghb, semiring, A, B) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, A, semiring, B) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, A, B, semiring) ; assert (isequal (C1, C2)) ;

C1 = gtb_mxm (ghb, semiring, a, b) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, a, semiring, b) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, a, b, semiring) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.mxm (semiring, A, B, desc)
%----------------------------------------------------------------------

% 2 matrices: A, B
% 1 string: semiring

C2 = A*B ;
c2 = a*b ;
assert (gbtest_err (c2, C2) < 1e-14) ;

C1 = gtb_mxm (ghb, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_mxm (ghb, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.mxm (C, accum, semiring, A, B, desc)
%----------------------------------------------------------------------

% 3 matrices: C, A, B
% 2 strings: accum, semiring

% C = accum (C, A*B) ;

C2 = C + A*B ;
c2 = c + a*b ;
assert (gbtest_err (c2, C2) < 1e-14) ;

C1 = gtb_mxm (ghb, C, accum, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, accum, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, accum, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, A, accum, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, A, accum, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, A, B, accum, semiring, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_mxm (ghb, c, accum, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, accum, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, accum, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, a, accum, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, a, accum, b, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, a, b, accum, semiring, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.mxm (C, M, semiring, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 1 string: semiring

% C<M> = A*B ;

T = A*B ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = a*b ;
c2 = c ;
c2 (m) = t (m) ;
assert (gbtest_err (c2, C2) < 1e-14) ;

C1 = gtb_mxm (ghb, semiring, C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, M, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, M, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, M, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_mxm (ghb, semiring, c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, m, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, m, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, m, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.mxm (C, M, accum, semiring, A, B, desc)
%----------------------------------------------------------------------

% 4 matrices: C, M, A, B
% 2 string: accum, semiring

% C<M> = accum (C, A*B) ;

T = C + A*B ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = c + a*b ;
c2 = c ;
c2 (m) = t (m) ;
assert (gbtest_err (c2, C2) < 1e-14) ;

C1 = gtb_mxm (ghb, C, M, accum, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, M, accum, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, M, accum, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, accum, semiring, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, accum, M, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, accum, M, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, accum, M, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, M, A, B, accum, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, M, A, accum, B, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, C, M, A, accum, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, accum, semiring, C, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, accum, C, semiring, M, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, accum, C, M, semiring, A, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, accum, C, M, A, semiring, B, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, accum, C, M, A, B, semiring, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_mxm (ghb, c, m, accum, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, m, accum, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, m, accum, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, accum, semiring, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, accum, m, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, accum, m, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, accum, m, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, m, a, b, accum, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, m, a, accum, b, semiring, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, c, m, a, accum, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, accum, semiring, c, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, accum, c, semiring, m, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, accum, c, m, semiring, a, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, accum, c, m, a, semiring, b, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_mxm (ghb, accum, c, m, a, b, semiring, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest86 (%d): all tests passed\n', ghb) ;

