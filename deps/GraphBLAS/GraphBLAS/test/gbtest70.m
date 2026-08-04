function gbtest70 (ghb)
%GBTEST70 test [GrB,GhB].random

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

% ghb cannot be 2 since it changes the state of the random
% number generator, which affects the resulting random matrix
% and causes some matrices to differ (C0 and C1 for example).
assert (ghb == 0 || ghb == 1) ;

rng ('default') ; A = sprand (4, 5, 0.5) ;
rng ('default') ; C0 = sprand (A) ;
rng ('default') ; C1 = gtb_random (ghb, A) ;
assert (isequal (C0, C1)) ;

types = gbtest_types ;

rng ('default') ;

for k = 1:length(types)
    type = types {k} ;

    rng ('default') ;
    G = gtb_random (ghb, 30, 40, 0.6) ; %#ok<*NASGU>

    r = gbtest_cast ([3 40], type) ;

    G = gtb_random (ghb, 300, 400, 0.6, 'range', r) ;
    assert (isequal (gtb_type (ghb, G), type)) ;

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

    G = gtb_random (ghb, 30, 40, 0.6, 'normal') ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;

    G = gtb_random (ghb, 30, 40, inf, 'normal') ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;
    assert (nnz (G) == prod (size (G))) ; %#ok<*PSIZE>

    G = gtb_random (ghb, 30, 40, 0.6, 'normal', 'range', r) ;
    assert (isequal (gtb_type (ghb, G), type)) ;

    G = gtb_random (ghb, 30, 40, 0.6, 'uniform') ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;

    G = gtb_random (ghb, 30, 40, 0.6, 'uniform', 'range', r) ;
    assert (isequal (gtb_type (ghb, G), type)) ;

    G = gtb_random (ghb, 30, 0.6, 'symmetric') ;
    assert (issymmetric (G)) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;

    G = gtb_random (ghb, 30, inf, 'symmetric') ;
    assert (issymmetric (G)) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;
    assert (nnz (G) == prod (size (G))) ;

    G = gtb_random (ghb, 30, 30, 0.6, 'unsymmetric') ;
    assert (~issymmetric (G)) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;

    G = gtb_random (ghb, 30, 0.6, 'normal', 'symmetric') ;
    assert (issymmetric (G)) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;

    G = gtb_random (ghb, 30, 0.6, 'normal', 'range', r, 'symmetric') ;
    assert (issymmetric (G)) ;
    assert (isequal (gtb_type (ghb, G), type)) ;

    G = gtb_random (ghb, 30, 0.6, 'normal', 'range', r, 'hermitian') ;
    assert (ishermitian (G)) ;
    assert (isequal (gtb_type (ghb, G), type)) ;

    S = sprandsym (30, 0.6) ;
    G = sprandsym (gtb (ghb, S)) ;
    assert (isequal (spones (G), spones (S))) ;
    assert (issymmetric (G)) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;

    S = sprandn (30, 40, 0.6) ;
    G = sprandn (gtb (ghb, S)) ;
    assert (isequal (spones (G), spones (S))) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;

    S = sprand (30, 40, 0.6) ;
    G = sprand (gtb (ghb, S)) ;
    assert (isequal (spones (G), spones (S))) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;

    G = sprand (10, 12, gtb (ghb, 0.5)) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;
    assert (gbtest_isa (ghb, G)) ;
    assert (isequal (size (G), [10 12])) ;

    G = sprandn (10, 12, gtb (ghb, 0.5)) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;
    assert (gbtest_isa (ghb, G)) ;
    assert (isequal (size (G), [10 12])) ;
    gnz = nnz (G) ;
    % nnz (G) is hard to predict because of duplicates
    assert (abs (10*12*0.5 - gnz) < 30) ;

    G = sprandn (10, 12, gtb (ghb, inf)) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;
    assert (gbtest_isa (ghb, G)) ;
    assert (isequal (size (G), [10 12])) ;
    assert (nnz (G) == 120) ;

    G = sprandsym (10, gtb (ghb, 0.5)) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;
    assert (gbtest_isa (ghb, G)) ;
    assert (isequal (size (G), [10 10])) ;
    gnz = nnz (G) ;
    assert (abs (10*10*0.5 - gnz) < 30) ;

    G = sprandsym (10, gtb (ghb, inf)) ;
    assert (isequal (gtb_type (ghb, G), 'double')) ;
    assert (gbtest_isa (ghb, G)) ;
    assert (isequal (size (G), [10 10])) ;
    assert (nnz (G) == 100)
    assert (gtb_isfull (ghb, G)) ;
    assert (gtb_isfull (ghb, double (G))) ;
    assert (gtb_isfull (ghb, full (G))) ;
    assert (gtb_isfull (ghb, full (double (G)))) ;

end

fprintf ('gbtest70 (%d): all tests passed\n', ghb) ;

