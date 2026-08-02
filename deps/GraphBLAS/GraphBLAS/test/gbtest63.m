function gbtest63 (ghb)
%GBTEST63 test [GrB,GhB].incidence

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

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
        [filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
        load (fullfile (filepath, './matrix/west0479_correct.mat')) ;
        W = Problem.A ;
    end

    W = spones (gtb_offdiag (ghb, W)) ;
    A = digraph (W) ;
    G = gtb (ghb, W) ;

    E0 = incidence (A) ;
    E1 = gtb_incidence (ghb, G) ;
    % E0 and E1 are the same, except the columns are in different orders
    E0 = sortrows (E0')' ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

    E1 = gtb_incidence (ghb, G, 'test_coverage') ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

    E1 = gtb_incidence (ghb, G, 'int8') ;
    assert (isequal (gtb_type (ghb, E1), 'int8')) ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

    W = W+W' ;
    A = graph (W) ;
    G = gtb (ghb, W) ;

    E0 = incidence (A) ;
    E1 = gtb_incidence (ghb, G, 'upper') ;
    E0 = sortrows (E0')' ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

    E1 = gtb_incidence (ghb, G, 'lower') ;
    E1 = -E1 ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

end

fprintf ('gbtest63 (%d): all tests passed\n', ghb) ;

