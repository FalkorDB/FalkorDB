function test31
%TEST31 test GrB_transpose

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n------------------- simple tests of GB_mex_transpose\n') ;

rng ('default') ;
A = sprand (4, 3, 0.4) ;
% full (A)
% full (A')
C = GB_mex_transpose (sparse (3,4), [ ], [ ], A) ;
assert (GB_spok (C.matrix) == 1) ;
assert (isequal (C.matrix,A')) ;

% C = A
D = struct ('inp0', 'tran') ;
C = GB_mex_transpose (sparse(4,3), [ ], [ ], A, D) ;
assert (isequal (C.matrix,A)) ;

% C = A'
C = GB_mex_transpose (sparse(3,4), [ ], [ ], A) ;
assert (isequal (C.matrix,A')) ;

Cin = sprand (4, 3, 0.5) ;
Cin2 = Cin' ;

% C = A'
C = GB_mex_transpose (Cin2, [ ], [ ], A) ;
assert (isequal (C.matrix,A')) ;

% C = Cin2+A'
C = GB_mex_transpose (Cin2, [ ], 'plus', A) ;
assert (isequal (C.matrix,Cin2+A')) ;

% C = Cin+A
D = struct ('inp0', 'tran') ;
C = GB_mex_transpose (Cin, [ ], 'plus', A, D) ;
assert (isequal (C.matrix,Cin+A)) ;

ops = {
    'first',
    'second',
    'pair',
    'oneb', % same as pair
    'min',
    'max',
    'plus',
    'minus',
    'times',
    'div',   } ;

for k = 1:length(ops)
    op = ops {k} ;

    % C = op (Cin2,A')
    D = struct ;
    C = GB_mex_transpose  (Cin2, [ ], op, A, D)  ;
    S = GB_spec_transpose (Cin2, [ ], op, A, D)  ;
    assert (isequal (C.matrix, sparse (S.matrix))) ;
    assert (isequal (GB_spones_mex (C.matrix), sparse (S.pattern))) ;

    % C = A', ignore the op
    D = struct ('outp', 'replace') ;
    C = GB_mex_transpose  (Cin2, [ ], op, A, D) ;
    S = GB_spec_transpose (Cin2, [ ], op, A, D) ;
    assert (isequal (GB_spones_mex (C.matrix), sparse (S.pattern))) ;

    % C = A, ignore the op
    D = struct ('inp0', 'tran', 'outp', 'replace') ;
    C = GB_mex_transpose  (Cin, [ ], op, A, D) ;
    S = GB_spec_transpose (Cin, [ ], op, A, D) ;
    assert (isequal (GB_spones_mex (C.matrix), sparse (S.pattern))) ;

    % C = op (Cin,A)
    D = struct ('inp0', 'tran') ;
    C = GB_mex_transpose  (Cin, [ ], op, A, D) ;
    S = GB_spec_transpose (Cin, [ ], op, A, D) ;
    assert (isequal (GB_spones_mex (C.matrix), sparse (S.pattern))) ;

end

%{
A = sprand (20, 16, 0.45)
C = GB_mex_transpose (A)

A = sparse (rand (40,10)) ;
C = GB_mex_transpose (A)
%}

fprintf ('\ntest31: all tests passed\n') ;

