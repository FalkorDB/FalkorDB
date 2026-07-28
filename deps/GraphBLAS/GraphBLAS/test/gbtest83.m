function gbtest83 (ghb, ghb2)
%GBTEST83 test [GrB,GhB].apply
%
% C = GrB.apply (op, A)
% C = GrB.apply (C, accum, op, A)
% C = GrB.apply (C, M, op, A)
% C = GrB.apply (C, M, accum, op, A)

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
op    = 'sqrt' ;
A     = gtb_random (ghb2, 9, 9, 0.5) ;
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

C1 = gtb_apply (ghb, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, A, op, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply (ghb, op, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, a, op, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply (C, accum, op, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 strings: accum, op

C2 = C + sqrt (A) ;
c2 = c + sqrt (a) ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply (ghb, C, accum, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, C, accum, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, C, A, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, accum, C, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, accum, C, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, accum, op, C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply (ghb, c, accum, op, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, c, accum, a, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, c, a, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, accum, c, a, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, accum, c, op, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, accum, op, c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply (C, M, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string:   op

% C<M> = sqrt (A)
C2 = gtb_assign (ghb, C, M, sqrt (A)) ;

t = sqrt (a) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply (ghb, C, M, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, C, M, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, C, op, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, op, C, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply (ghb, c, m, op, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, c, m, a, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, c, op, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, op, c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.apply (C, M, accum, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 strings:  accum, op

% C<M> += sqrt (A)
C2 = gtb_assign (ghb, C, M, accum, sqrt (A)) ;

t = c + sqrt (a) ;
c2 = c ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_apply (ghb, C, M, accum, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, C, M, accum, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, C, M, A, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, C, accum, M, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, C, accum, M, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, C, accum, op, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_apply (ghb, accum, C, M, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, accum, C, M, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, accum, C, op, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_apply (ghb, accum, op, C, M, A, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest83 (%d): all tests passed\n', ghb) ;

