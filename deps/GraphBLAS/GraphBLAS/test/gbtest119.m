function gbtest119 (ghb, ghb2)
%GBTEST119 test [GrB,GhB].eunion
%
% C = GrB.eunion (op, A, alpha, B, beta)
% C = GrB.eunion (op, A, alpha, B, beta, desc)
% C = GrB.eunion (C, accum, op, A, alpha, B, beta, desc)
% C = GrB.eunion (C, M, op, A, alpha, B, beta, desc)
% C = GrB.eunion (C, M, accum, op, A, alpha, B, beta, desc)

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
A     = gtb_random (ghb2, 9, 9, 0.5) ;
B     = gtb_random (ghb2, 9, 9, 0.5) ;
desc  = struct ;

op = '-' ;
alpha = 0 ;
beta = 0 ;

c = double (C) ;
m = logical (M) ;
a = double (A) ;
b = double (B) ;

%----------------------------------------------------------------------
% C = GrB.eunion (op, A, alpha, B, beta)
%----------------------------------------------------------------------

% 4 matrices: A, alpha, B, beta
% 1 string: op

C2 = A-B ;
c2 = a-b ;
assert (isequal (c2, C2)) ;

C1 = gtb_eunion (ghb, op, A, alpha, B, beta) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, A, alpha, op, B, beta) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, A, alpha, B, beta, op) ; assert (isequal (C1, C2)) ;

C1 = gtb_eunion (ghb, op, a, alpha, b, beta) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, a, alpha, op, b, beta) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, a, alpha, b, beta, op) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eunion (op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 4 matrices: A, alpha, B, beta
% 1 string: op

C2 = A-B ;
c2 = a-b ;
assert (isequal (c2, C2)) ;

C1 = gtb_eunion (ghb, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_eunion (ghb, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eunion (C, accum, op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 5 matrices: C, A, alpha, B, beta
% 2 strings: accum, op

% C = accum (C, op (A,B)) ;

C2 = C + (A-B) ;
c2 = c + (a-b) ;
assert (isequal (c2, C2)) ;

C1 = gtb_eunion (ghb, C, accum, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, accum, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, accum, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, A, alpha, accum, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, A, alpha, accum, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, A, alpha, B, beta, accum, op, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_eunion (ghb, c, accum, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, accum, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, accum, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, a, alpha, accum, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, a, alpha, accum, b, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, a, alpha, b, beta, accum, op, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_eunion (ghb, C, accum, op, A, alpha, B, beta      ) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, accum, A, alpha, op, B, beta      ) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, accum, A, alpha, B, beta, op      ) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, A, alpha, accum, op, B, beta      ) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, A, alpha, accum, B, beta, op      ) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, A, alpha, B, beta, accum, op      ) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eunion (C, M, op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 6 matrices: C, M, A, alpha, B, beta
% 1 string: op

% C<M> = op (A,B)

T = (A-B) ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = (a-b) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_eunion (ghb, op, C, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, M, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, M, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, M, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_eunion (ghb, op, c, m, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, m, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, m, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, m, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.eunion (C, M, accum, op, A, alpha, B, beta, desc)
%----------------------------------------------------------------------

% 6 matrices: C, M, A, alpha, B, beta
% 2 string: accum, op

% C<M> = accum (C, A*B) ;

T = C + (A-B) ;
C2 = gtb (ghb, C) ;
C2 (M) = T (M) ;

t = c + (a-b) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_eunion (ghb, C, M, accum, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, M, accum, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, M, accum, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, accum, op, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, accum, M, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, accum, M, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, accum, M, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, M, A, alpha, B, beta, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, M, A, alpha, accum, B, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, C, M, A, alpha, accum, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, accum, op, C, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, accum, C, op, M, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, accum, C, M, op, A, alpha, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, accum, C, M, A, alpha, op, B, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, accum, C, M, A, alpha, B, beta, op, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_eunion (ghb, c, m, accum, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, m, accum, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, m, accum, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, accum, op, m, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, accum, m, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, accum, m, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, accum, m, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, m, a, alpha, b, beta, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, m, a, alpha, accum, b, beta, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, c, m, a, alpha, accum, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, accum, op, c, m, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, accum, c, op, m, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, accum, c, m, op, a, alpha, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, accum, c, m, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_eunion (ghb, accum, c, m, a, alpha, b, beta, op, desc) ; assert (isequal (C1, C2)) ;

beta = gtb (ghb2, beta) ;
alpha = gtb (ghb2, alpha) ;
C1 = gtb_eunion (ghb, accum, c, m, a, alpha, op, b, beta, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest119 (%d): all tests passed\n', ghb) ;

