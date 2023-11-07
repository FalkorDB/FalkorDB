function gbtest49
%GBTEST49 test GrB.prune

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

for trial = 1:40

    A = rand (4) ;
    A (A > .5) = 0 ;
    A (1,1) = 1 ;
    G = GrB (A) ;

    C0 = sparse (A) ;
    C1 = GrB.prune (A) ;
    C2 = GrB.prune (G) ;
    assert (isequal (C0, C1)) ;
    assert (isequal (C0, C2)) ;

    C0 = sparse (A) ;
    C0 (1,1) = 0 ; %#ok<*SPRIX>
    C1 = GrB.prune (A, 1) ;
    C2 = GrB.prune (G, 1) ;
    assert (isequal (C0, double (C1))) ;
    assert (isequal (C0, double (C2))) ;

end

fprintf ('gbtest49: all tests passed\n') ;

