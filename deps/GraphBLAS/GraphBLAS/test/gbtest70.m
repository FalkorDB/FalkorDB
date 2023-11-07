function gbtest70
%GBTEST70 test GrB.random

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ; A = sprand (4, 5, 0.5) ;
rng ('default') ; C0 = sprand (A) ;
rng ('default') ; C1 = GrB.random (A) ;
assert (isequal (C0, C1)) ;

types = gbtest_types ;

rng ('default') ;

for k = 1:length(types)
    type = types {k} ;

    rng ('default') ;
    G = GrB.random (30, 40, 0.6) ; %#ok<*NASGU>

    r = gbtest_cast ([3 40], type) ;
    G = GrB.random (300, 400, 0.6, 'range', r) ;
    assert (isequal (GrB.type (G), type)) ;

    if (~isequal (type, 'logical'))
        [i,j,x] = find (G) ; %#ok<*ASGLU>
        if (isinteger (r))
            assert (min (r) == min (r)) ;
            assert (max (r) == max (r)) ;
        elseif (isreal (r))
            d = min (x) - min (r) ; assert (d > 0 && d < 0.01) ;
            d = max (r) - max (x) ; assert (d > 0 && d < 0.01) ;
        end
    end

    G = GrB.random (30, 40, 0.6, 'normal') ;
    assert (isequal (GrB.type (G), 'double')) ;

    G = GrB.random (30, 40, inf, 'normal') ;
    assert (isequal (GrB.type (G), 'double')) ;
    assert (nnz (G) == prod (size (G))) ; %#ok<*PSIZE>

    G = GrB.random (30, 40, 0.6, 'normal', 'range', r) ;
    assert (isequal (GrB.type (G), type)) ;

    G = GrB.random (30, 40, 0.6, 'uniform') ;
    assert (isequal (GrB.type (G), 'double')) ;

    G = GrB.random (30, 40, 0.6, 'uniform', 'range', r) ;
    assert (isequal (GrB.type (G), type)) ;

    G = GrB.random (30, 0.6, 'symmetric') ;
    assert (issymmetric (G)) ;
    assert (isequal (GrB.type (G), 'double')) ;

    G = GrB.random (30, inf, 'symmetric') ;
    assert (issymmetric (G)) ;
    assert (isequal (GrB.type (G), 'double')) ;
    assert (nnz (G) == prod (size (G))) ;

    G = GrB.random (30, 30, 0.6, 'unsymmetric') ;
    assert (~issymmetric (G)) ;
    assert (isequal (GrB.type (G), 'double')) ;

    G = GrB.random (30, 0.6, 'normal', 'symmetric') ;
    assert (issymmetric (G)) ;
    assert (isequal (GrB.type (G), 'double')) ;

    G = GrB.random (30, 0.6, 'normal', 'range', r, 'symmetric') ;
    assert (issymmetric (G)) ;
    assert (isequal (GrB.type (G), type)) ;

    G = GrB.random (30, 0.6, 'normal', 'range', r, 'hermitian') ;
    assert (ishermitian (G)) ;
    assert (isequal (GrB.type (G), type)) ;

    S = sprandsym (30, 0.6) ;
    G = sprandsym (GrB (S)) ;
    assert (isequal (spones (G), spones (S))) ;
    assert (issymmetric (G)) ;
    assert (isequal (GrB.type (G), 'double')) ;

    S = sprandn (30, 40, 0.6) ;
    G = sprandn (GrB (S)) ;
    assert (isequal (spones (G), spones (S))) ;
    assert (isequal (GrB.type (G), 'double')) ;

    S = sprand (30, 40, 0.6) ;
    G = sprand (GrB (S)) ;
    assert (isequal (spones (G), spones (S))) ;
    assert (isequal (GrB.type (G), 'double')) ;

    G = sprand (10, 12, GrB (0.5)) ;
    assert (isequal (GrB.type (G), 'double')) ;
    assert (isa (G, 'GrB')) ;
    assert (isequal (size (G), [10 12])) ;

    G = sprandn (10, 12, GrB (0.5)) ;
    assert (isequal (GrB.type (G), 'double')) ;
    assert (isa (G, 'GrB')) ;
    assert (isequal (size (G), [10 12])) ;
    gnz = nnz (G) ;
    % nnz (G) is hard to predict because of duplicates
    assert (abs (10*12*0.5 - gnz) < 30) ;

    G = sprandn (10, 12, GrB (inf)) ;
    assert (isequal (GrB.type (G), 'double')) ;
    assert (isa (G, 'GrB')) ;
    assert (isequal (size (G), [10 12])) ;
    assert (nnz (G) == 120) ;

    G = sprandsym (10, GrB (0.5)) ;
    assert (isequal (GrB.type (G), 'double')) ;
    assert (isa (G, 'GrB')) ;
    assert (isequal (size (G), [10 10])) ;
    gnz = nnz (G) ;
    assert (abs (10*10*0.5 - gnz) < 30) ;

    G = sprandsym (10, GrB (inf)) ;
    assert (isequal (GrB.type (G), 'double')) ;
    assert (isa (G, 'GrB')) ;
    assert (isequal (size (G), [10 10])) ;
    assert (nnz (G) == 100)
    assert (GrB.isfull (G)) ;
    assert (GrB.isfull (double (G))) ;
    assert (GrB.isfull (full (G))) ;
    assert (GrB.isfull (full (double (G)))) ;

end


fprintf ('gbtest70: all tests passed\n') ;

