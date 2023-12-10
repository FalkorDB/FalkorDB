function gbtest39
%GBTEST39 test amd, colamd, symamd, symrcm, dmperm, etree

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

for trial = 1:40
    fprintf ('.') ;

    n = 20 ;
    A = sprand (n, n, 0.1) ;
    S = A + A' ;
    G = GrB (A) ;
    H = GrB (S) ;

    assert (isequal (amd (A),    amd (G))) ;
    assert (isequal (amd (S),    amd (H))) ;

    assert (isequal (colamd (A), colamd (G))) ;
    assert (isequal (colamd (S), colamd (H))) ;

    assert (isequal (symamd (A), symamd (G))) ;
    assert (isequal (symamd (S), symamd (H))) ;

    assert (isequal (symrcm (A), symrcm (G))) ;
    assert (isequal (symrcm (S), symrcm (H))) ;

    assert (isequal (etree (A), etree (G))) ;
    assert (isequal (etree (S), etree (H))) ;

    [p1, post1] = dmperm (A) ;
    [p2, post2] = dmperm (G) ;
    assert (isequal (p1, p2)) ;
    assert (isequal (post1, post2)) ;

    [p1, post1] = dmperm (S) ;
    [p2, post2] = dmperm (H) ;
    assert (isequal (p1, p2)) ;
    assert (isequal (post1, post2)) ;

    assert (isequal (dmperm (A), dmperm (G))) ;
    assert (isequal (dmperm (S), dmperm (H))) ;

    [p1, q1, r1, s1, cc1, rr1] = dmperm (A) ;
    [p2, q2, r2, s2, cc2, rr2] = dmperm (G) ;
    assert (isequal (p1, p2)) ;
    assert (isequal (q1, q2)) ;
    assert (isequal (r1, r2)) ;
    assert (isequal (s1, s2)) ;
    assert (isequal (cc1, cc2)) ;
    assert (isequal (rr1, rr2)) ;

    [p1, q1, r1, s1, cc1, rr1] = dmperm (S) ;
    [p2, q2, r2, s2, cc2, rr2] = dmperm (H) ;
    assert (isequal (p1, p2)) ;
    assert (isequal (q1, q2)) ;
    assert (isequal (r1, r2)) ;
    assert (isequal (s1, s2)) ;
    assert (isequal (cc1, cc2)) ;
    assert (isequal (rr1, rr2)) ;

end

fprintf ('\ngbtest39: all tests passed\n') ;
