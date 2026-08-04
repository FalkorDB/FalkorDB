function gbtest49 (ghb)
%GBTEST49 test [GrB,GhB].prune

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

for trial = 1:40

    A = rand (4) ;
    A (A > .5) = 0 ;
    A (1,1) = 1 ;
    G = gtb (ghb, A) ;

    C0 = sparse (A) ;
    C1 = gtb_prune (ghb, A) ;
    C2 = gtb_prune (ghb, G) ;
    assert (isequal (C0, C1)) ;
    assert (isequal (C0, C2)) ;

    C0 = sparse (A) ;
    C0 (1,1) = 0 ; %#ok<*SPRIX>
    C1 = gtb_prune (ghb, A, 1) ;
    C2 = gtb_prune (ghb, G, 1) ;
    assert (isequal (C0, double (C1))) ;
    assert (isequal (C0, double (C2))) ;

end

fprintf ('gbtest49 (%d): all tests passed\n', ghb) ;

