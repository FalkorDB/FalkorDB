function gbtest63
%GBTEST63 test GrB.incidence

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

for trial = 1:2

    if (trial == 1)
        ij = [
        4 1
        1 2
        4 3
        6 3
        7 3
        1 4
        7 4
        2 5
        7 5
        3 6
        5 6
        2 7 ] ;
        W = sparse (ij (:,1), ij (:,2), ones (12,1), 8, 8) ;
    else
        % load west0479 ; %#ok<*LOAD>
        % W = west0479 ;
        load west0479_correct ; %#ok<*LOAD>
        W = Problem.A ;
    end

    W = spones (GrB.offdiag (W)) ;
    A = digraph (W) ;
    G = GrB (W) ;

    E0 = incidence (A) ;
    E1 = GrB.incidence (G) ;
    % E0 and E1 are the same, except the columns are in different orders
    E0 = sortrows (E0')' ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

    E1 = GrB.incidence (G, 'int8') ;
    assert (isequal (GrB.type (E1), 'int8')) ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

    W = W+W' ;
    A = graph (W) ;
    G = GrB (W) ;

    E0 = incidence (A) ;
    E1 = GrB.incidence (G, 'upper') ;
    E0 = sortrows (E0')' ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

    E1 = GrB.incidence (G, 'lower') ;
    E1 = -E1 ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

end

fprintf ('gbtest63: all tests passed\n') ;

