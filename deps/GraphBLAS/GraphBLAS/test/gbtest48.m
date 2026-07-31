function gbtest48 (ghb)
%GBTEST48 test [GrB,GhB].apply

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

desc.kind = 'sparse' ;

for trial = 1:40

    A = rand (4) ;
    A (A > .5) = 0 ;
    G = gtb (ghb, A) ;

    C0 = -A ;
    C1 = gtb_apply (ghb, 'negate', A) ;
    C2 = gtb_apply (ghb, 'negate', A, desc) ;
    C3 = gtb_apply (ghb, 'negate', G, desc) ;
    C4 = gtb_apply (ghb, 'negate', G) ;

    assert (isequal (C0, C1)) ;
    assert (isequal (C0, C2)) ;
    assert (isequal (C0, C3)) ;
    assert (isequal (C0, C4)) ;

    assert (isequal (class (C2), 'double')) ;
    assert (isequal (class (C3), 'double')) ;

    M = logical (sprand (4, 4, 0.5)) ;
    Cin = rand (4) ;
    T = Cin + (-A) ;
    C0 = Cin ;
    C0 (M) = T (M) ;
    C1 = gtb_apply (ghb, Cin, M, '+', '-', A) ;
    assert (isequal (C0, C1)) ;

    C0 = Cin + (-A) ;
    C1 = gtb_apply (ghb, Cin, '+', '-', A) ;
    assert (isequal (C0, C1)) ;

    T = -A ;
    C0 = Cin ;
    C0 (M) = T (M) ;
    C1 = gtb_apply (ghb, Cin, M, '', '-', A) ;
    assert (isequal (C0, C1)) ;

    N = logical (sprand (4, 4, 0.5)) ;
    C0 = M & N ;
    C1 = gtb_emult (ghb, '&', M, N) ;
    d.kind = 'builtin' ;
    d.format = 'by row' ;
    C2 = gtb_emult (ghb, '&', ...
        gtb (ghb, M, 'by row'), gtb (ghb, N, 'by row'), d) ;
    assert (isequal (C0, C1)) ;
    assert (isequal (C0, C2)) ;

end

fprintf ('gbtest48 (%d): all tests passed\n', ghb) ;

