function gbtest134
%GBTEST134 test GhB.apply (inplace usage only)
%
% GhB.apply (C, unop)                      % C = unop (C)
% GhB.apply (C, accum, unop)               % C += unop (C)
% GhB.apply (C, unop, A)                   % C = unop (A)
% GhB.apply (C, accum, unop, A)            % C += unop (A)
% GhB.apply (C, M, unop, A)                % C<M> = unop (A)
% GhB.apply (C, M, accum, unop, A)         % C<M> += unop (A)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

C     = GhB.random (9, 9, 0.5) ;
M     = GhB.random (9, 9, 0.5, 'range', logical ([false true])) ;
accum = '+' ;
op    = 'sqrt' ;
A     = GhB.random (9, 9, 0.5) ;
desc  = struct ;
C0    = GhB (9, 9) ;

%----------------------------------------------------------------------
% GhB.apply (C, op, C, desc)
%----------------------------------------------------------------------

% 2 matrices: C twice
% 1 string: op

C2 = sqrt (C) ;
C3 = GhB.apply (C0, op, C, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply (C1, op, C1, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (op, C1, C1, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, C1, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GhB (C) ; GhB.apply (C1, op, C1) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (op, C1, C1) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, C1, op) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply (C, op, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 1 string: op

C2 = sqrt (A) ;
C3 = GhB.apply (C0, op, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply (C1, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (op, C1, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, A, op, desc) ; assert (isequal (C1, C2)) ;

C1 = GhB (C) ; GhB.apply (C1, op, A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (op, C1, A) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, A, op) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply (C, accum, op, A, desc)
%----------------------------------------------------------------------

% 2 matrices: C, A
% 2 strings: accum, op

C2 = C + sqrt (A) ;
C3 = GhB.apply (C, accum, op, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply (C1, accum, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, accum, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, A, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (accum, C1, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (accum, C1, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (accum, op, C1, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply (C, M, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 1 string:   op

% C<M> = sqrt (A)
C2 = GhB.assign (C, M, sqrt (A)) ;
C3 = GhB.apply (C, M, op, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply (C1, M, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, M, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, op, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (op, C1, M, A, desc) ; assert (isequal (C1, C2)) ;

%----------------------------------------------------------------------
% GhB.apply (C, M, accum, op, A, desc)
%----------------------------------------------------------------------

% 3 matrices: C, M, A
% 2 strings:  accum, op

% C<M> += sqrt (A)
C2 = GhB.assign (C, M, accum, sqrt (A)) ;
C3 = GhB.apply (C, M, accum, op, A, desc) ;
assert (isequal (C2, C3)) ;

C1 = GhB (C) ; GhB.apply (C1, M, accum, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, M, accum, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, M, A, accum, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, accum, M, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, accum, M, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (C1, accum, op, M, A, desc) ; assert (isequal (C1, C2)) ;

C1 = GhB (C) ; GhB.apply (accum, C1, M, op, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (accum, C1, M, A, op, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (accum, C1, op, M, A, desc) ; assert (isequal (C1, C2)) ;
C1 = GhB (C) ; GhB.apply (accum, op, C1, M, A, desc) ; assert (isequal (C1, C2)) ;

fprintf ('gbtest134: all tests passed\n') ;

