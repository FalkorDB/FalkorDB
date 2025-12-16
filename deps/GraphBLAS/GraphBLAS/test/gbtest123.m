function gbtest123
%GBTEST123 test build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default')

n = 1000 ;
H = GrB (n, n) ;
H (1,1) = 1 ;
S = GrB.build (H,H,pi) ;
P = sparse (pi) ;
assert (isequal (S, P)) ;

n = flintmax ;
H = GrB (n, n) ;
H (1,1) = 1 ;
try
    S = GrB.build (H,H,H) ;
    ok = false ;
catch me
    me
    have_octave = gb_octave ;
    if (have_octave)
        assert (isequal (me.message, 'gbbuild: input matrix dimensions are too large')) ;
    else
        assert (isequal (me.message, 'input matrix dimensions are too large')) ;
    end
    ok = true ;
end
assert (ok) ;

fprintf ('\ngbtest123: all tests passed\n') ;
