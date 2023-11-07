function gbtest29
%GBTEST29 test subsref and subsasgn with logical indexing

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
have_octave = gb_octave ;

types = gbtest_types ;
for k = 1:length (types)
    type = types {k} ;
    C = gbtest_cast (magic (3), type) ;
    M = logical (C > 3) ;
    A = gbtest_cast (2 * magic (3), type) ;
    C (M) = A (M) ;
    G = GrB (magic (3), type) ;
    G (M) = A (M) ;
    assert (gbtest_err (G, C) == 0) ;
    if (isreal (C) && isreal (G))
        assert (gbtest_eq (G, C)) ;
    else
        assert (gbtest_eq (real (G), real (C))) ;
        assert (gbtest_eq (imag (G), imag (C))) ;
    end

    C0 = GrB.random (3, 3, inf, 'range', GrB ([0 1], type)) ;
    M = true (3) ;
    A = GrB (1:9, type) ;
    C1 = C0 ;
    C1 (M) = A ;
    C2 = double (C0) ;
    C2 (M) = double (A) ;
    assert (isequal (double (C1), C2)) ;

    A = GrB (A, 'by row') ;
    C1 = C0 ;
    C1 (M) = A ;
    assert (isequal (double (C1), C2)) ;
    A = GrB (A', 'by row') ;
    C1 = C0 ;
    C1 (M) = A ;
    assert (isequal (double (C1), C2)) ;
end

for trial = 1:40
    fprintf ('.')
    for m = 0:5
        for n = 0:5
            A = sprand (m, n, 0.5) ;
            C = sprand (m, n, 0.5) ;
            M = sprand (m, n, 0.5) ~= 0 ;
            G = GrB (A) ;

            x1 = A (M) ; %#ok<*NASGU>
            x2 = G (M) ;

            % With built-in vectors, if A and M are row vectors then A(M)
            % is also a row vector.

            C1 = C ;
            C1 (M) = A (M) ;%#ok<*SPRIX> % C1(M) builtin, A(M) is built-in

            C2 = GrB (C) ;
            C2 (M) = A (M) ;        % C2(M) is GrB, A(M) is built-in

            C3 = GrB (C) ;
            C3 (M) = G (M) ;        % C3(M) is GrB, and G(M) is GrB
            assert (gbtest_eq (C1, C2)) ;
            assert (gbtest_eq (C1, C3)) ;

            % using the built-in subsasgn
            C4 = C ;
            if (have_octave)
                % Octave does not do the auto typecast.
                C4 (M) = double (G (M)) ;
            else
                % This uses the built-in subsasgn, after typecasting G(M) from
                % class GrB to class double, using GrB/double.  MATLAB does the
                % automatic typecasting of G(M), since it sees that GrB has a
                % "double" method.
                C4 (M) = G (M) ;
            end
            assert (gbtest_eq (C1, C4)) ;

            % test assignment with A iso 
            G = spones (GrB (A)) ;
            A = double (G) ;
            C1 = C ;
            C1 (M) = A (M) ;%#ok<*SPRIX> % C1(M) builtin, A(M) is built-in
            C2 = GrB (C) ;
            C2 (M) = G (M) ;
            assert (gbtest_eq (C1, C2)) ;

            % also try with a GrB mask matrix M
            M = GrB (M) ;
            C5 = GrB (C) ;
            C5 (M) = G (M) ;
            assert (gbtest_eq (C1, C5)) ;

            % test scalar assigment with logical indexing
            K = logical (M) ;
            C1 (K) = pi ;
            C2 (M) = pi ;
            C3 (M) = GrB (pi) ;
            if (have_octave)
                % See above for the Octave vs MATLAB difference in casting.
                C4 (K) = double (GrB (pi)) ;
            else
                C4 (K) = GrB (pi) ;
            end
            assert (gbtest_eq (C1, C4)) ;
            C5 (M) = pi ;
            assert (gbtest_eq (C1, C2)) ;
            assert (gbtest_eq (C1, C3)) ;
            assert (gbtest_eq (C1, C5)) ;
        end
    end
end

fprintf ('\ngbtest29: all tests passed\n') ;

