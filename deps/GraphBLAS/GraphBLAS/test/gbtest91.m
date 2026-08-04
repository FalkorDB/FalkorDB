function gbtest91 (ghb, ghb2)
%GBTEST91 test [GrB,GhB].trans
%
% C = GrB.trans (A)
% C = GrB.trans (A, desc)
% C = GrB.trans (C, accum, A, desc)
% C = GrB.trans (C, M, A, desc)
% C = GrB.trans (C, M, accum, A, desc)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
if (nargin < 2)
    ghb2 = ghb ;
end
gtb_name = gtb_prep (ghb) ;

C      = gtb_random (ghb2, 8, 9, 0.5) ;
M      = gtb_random (ghb2, 8, 9, 0.5, 'range', logical ([false true])) ;
accum  = '+' ;
A      = gtb_random (ghb2, 9, 8, 0.5) ;
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

C1 = gtb_trans (ghb, A) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, a) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.trans (A, desc)
%----------------------------------------------------------------------

% 1 matrix: A
% 0 string:

C2 = A.' ;
c2 = a.' ;
assert (isequal (c2, C2)) ;

C1 = gtb_trans (ghb, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.trans (C, accum, A, desc)
%----------------------------------------------------------------------

% 2 matrices C, A
% 1 string: accum

C2 = C + A.' ;
c2 = c + a.' ;
assert (isequal (c2, C2)) ;

C1 = gtb_trans (ghb, C, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, C, A, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, accum, C, A, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_trans (ghb, c, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, c, a, accum, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, accum, c, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.trans (C, M, A, desc)
%----------------------------------------------------------------------

% 3 matrices C, M, A

% C<M> = A.'

C2 = gtb (ghb, C) ;
T = A.' ;
C2 (M) = T (M) ;

c2 = c ;
t = a.' ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_trans (ghb, C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, c, m, a, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% C = GrB.trans (C, M, accum, A, desc)
%----------------------------------------------------------------------

% 3 matrices C, M, A
% 1 string: accum

% C<M> += A.'

C2 = gtb (ghb, C) ;
T = C + A.' ;
C2 (M) = T (M) ;

c2 = c ;
t = c + a.' ;
c2 (m) = t (m) ;
assert (isequal (c2, C2)) ;

C1 = gtb_trans (ghb, C, M, accum, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, accum, C, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, C, accum, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, C, M, A, accum, desc) ; assert (isequal (C1, C2)) ;

C1 = gtb_trans (ghb, c, m, accum, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, accum, c, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, c, accum, m, a, desc) ; assert (isequal (C1, C2)) ;
C1 = gtb_trans (ghb, c, m, a, accum, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest91 (%d): all tests passed\n', ghb) ;

