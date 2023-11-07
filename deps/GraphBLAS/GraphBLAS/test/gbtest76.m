function gbtest76
%GBTEST76 test trig and other functions

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ngbtest76: testing trig and special functions\n') ;

rng ('default') ;

types = { 'single', 'double', ...
    'single complex', 'double complex', 'int32', 'uint32' } ;

for k = 1:length (types)
    type = types {k} ;

    fprintf ('%s\n', type) ;

    switch (type)
        case { 'single' }
            A = single (rand (4)) ;
            B = single (rand (4)) ;
            tol = 1e-5 ;
            G = GrB (A) ;
            H = GrB (B) ;
            gbtest76b (A, B, G, H, tol) ;
        case { 'double' }
            A = rand (4) ;
            B = rand (4) ;
            tol = 1e-10 ;
            G = GrB (A) ;
            H = GrB (B) ;
            gbtest76b (A, B, G, H, tol) ;
        case { 'single complex' }
            A = single (rand (4) + 1i* rand (4)) ;
            B = single (rand (4) + 1i* rand (4)) ;
            tol = 1e-5 ;
            G = GrB (A) ;
            H = GrB (B) ;
            gbtest76b (A, B, G, H, tol) ;
        case { 'double complex' }
            A = rand (4) + 1i* rand (4) ;
            B = rand (4) + 1i* rand (4) ;
            tol = 1e-10 ;
            G = GrB (A) ;
            H = GrB (B) ;
            gbtest76b (A, B, G, H, tol) ;
        case { 'int32' }
            A = int32 (magic (4)) ;
            B = int32 (rand (4) * 4) ;
            tol = 1e-10 ;
            G = GrB (A) ;
            H = GrB (B) ;
            gbtest76b (double (A), double (B), G, H, tol) ;
        case { 'uint32' }
            A = uint32 (magic (4)) ;
            B = uint32 (rand (4) * 4) ;
            tol = 1e-10 ;
            G = GrB (A) ;
            H = GrB (B) ;
            gbtest76b (double (A), double (B), G, H, tol) ;
    end
end

GrB.finalize ;
fprintf ('\ngbtest76: all tests passed\n') ;
end

function gbtest76b (A, B, G, H, tol)

    have_octave = gb_octave ;

    A (1,1) = 0 ;
    G (1,1) = 0 ;
    G = GrB.prune (G) ;
    S = spones (G) ;

    C1 = hypot (A, B) ;
    C2 = hypot (G, H) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = atan2 (real (A), real (B)) ;
    C2 = atan2 (real (G), real (H)) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = atan2 (pi, real (B)) ;
    C2 = atan2 (pi, real (H)) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = atan2 (real (A), 3) ;
    C2 = atan2 (real (G), 3) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = atan2 (pi, 3) ;
    C2 = atan2 (GrB (pi), 3) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = atan2 (0, 3) ;
    C2 = atan2 (real (GrB (1,1,GrB.type(G))), 3) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = complex (real (A), real (B)) ;
    C2 = complex (real (G), real (H)) ;
    err = norm (double (C1) - double (C2), 1) ;
    assert (err < tol) ;

    C1 = complex (pi, real (B)) ;
    C2 = complex (pi, real (H)) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = complex (real (A), 3) ;
    C2 = complex (real (G), 3) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = complex (real (A), 0) ;
    C2 = complex (real (G), 0) ;
    err = norm (double (C1) - double (C2), 1) ;
    assert (err < tol) ;

    C1 = complex (0, real (A)) ;
    C2 = complex (0, real (G)) ;
    err = norm (double (C1) - double (C2), 1) ;
    assert (err < tol) ;

    C1 = complex (pi, 3) ;
    C2 = complex (GrB (pi), 3) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = complex (0, 3) ;
    C2 = complex (real (GrB (1,1,GrB.type(G))), 3) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = acosh (A) ;
    C2 = acosh (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = acos (A) ;
    C2 = acos (G) ;
    err = norm (S .* (cos (C1) - cos (C2)), 1) ;
    assert (err < tol) ;

    C1 = acoth (A) ;
    C2 = acoth (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = acot (A) ;
    C2 = acot (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = acsch (A) ;
    C2 = acsch (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = acsc (A) ;
    if (isreal (C1) || ~ispc)
        % Windows casinf and casin are broken
        C2 = acsc (G) ;
        err = norm (csc (C1) - csc (C2),1) ;
        assert (err < tol) ;
    end

    C1 = angle (A) ;
    C2 = angle (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = asech (A) ;
    C2 = asech (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = asec (A) ;
    C2 = asec (G) ;
    err = norm (sec (C1) - sec (C2),1) ;
    assert (err < tol) ;

    C1 = asinh (A) ;
    C2 = asinh (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = asin (A) ;
    if (isreal (C1) || ~ispc)
        % Windows casinf and casin are broken
        C2 = asin (G) ;
        err = norm (sin (C1) - sin (C2), 1) ;
        assert (err < tol) ;
    end

    C1 = asin (2*A) ;
    if (isreal (C1) || ~ispc)
        % Windows casinf and casin are broken
        C2 = asin (2*G) ;
        err = norm (sin (C1) - sin (C2), 1) ;
        assert (err < tol) ;
    end

    C1 = atanh (A) ;
    C2 = atanh (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = atan (A) ;
    C2 = atan (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = conj (A) ;
    C2 = conj (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = cosh (A) ;
    C2 = cosh (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = cos (A) ;
    C2 = cos (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    if (gb_contains (GrB.type (G), 'double'))
        % built-in methods cannot compute spfun for single;
        % built-in methods and GraphBLAS can't do this for int*
        C1 = spfun ('cos', A) ;
        C2 = spfun ('cos', G) ;
        err = norm (C1-C2, 1) ;
        assert (err < tol) ;

        C1 = spfun (@cos, A) ;
        C2 = spfun (@cos, G) ;
        err = norm (C1-C2, 1) ;
        assert (err < tol) ;

        % GraphBLAS doesn't have cosd, so it falls
        % through to use feval('cosd',x).
        C1 = spfun ('cosd', A) ;
        C2 = spfun ('cosd', G) ;
        err = norm (C1-C2, 1) ;
        assert (err < tol) ;
    end

    C1 = coth (A) ;
    C2 = coth (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = cot (A) ;
    C2 = cot (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = csch (A) ;
    C2 = csch (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = csc (A) ;
    C2 = csc (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = erfc (real (full (A))) ;
    C2 = erfc (real (G)) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = erf (real (full (A))) ;
    C2 = erf (real (G)) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = nthroot (real (A), 3) ;
    C2 = cbrt (real (G)) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = exp (A) ;
    C2 = exp (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = expm1 (A) ;
    C2 = expm1 (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = gammaln (real (full (A))) ;
    C2 = gammaln (real (G)) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = gamma (real (full (A))) ;
    C2 = gamma (real (G)) ;
    assert (isinf (C1 (1,1)) == isinf (C2 (1,1)))
    C1 (1,1) = 0 ;
    C2 (1,1) = 0 ;
    err = norm (C1-C2, 1) / norm (C1, 1) ;
    assert (err < tol) ;

    C1 = imag (A) ;
    C2 = imag (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = log10 (A) ;
    C2 = log10 (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    if (~have_octave || isreal (A))
        % octave7: log1p(0) is 0 which is correct, but log1p(0+0i) is
        % (0+1.5708i), even though log(complex(1)) is 0.
        C1 = log1p (A) ;
        C2 = log1p (G) ;
        err = norm (C1-C2, 1) ;
        assert (err < tol) ;
    end

    C1 = log2 (A) ;
    C2 = log2 (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    [F1,E1] = log2 (real (A)) ;
    [F2,E2] = log2 (G) ;
    C1 = F1 .* (2 .^ E1) ;
    C2 = F2 .* (2 .^ E2) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = pow2 (F1, E1) ;
    C2 = pow2 (F2, E2) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = log (A) ;
    C2 = log (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = pow2 (A) ;
    C2 = pow2 (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = sech (A) ;
    C2 = sech (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = sec (A) ;
    C2 = sec (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = sinh (A) ;
    C2 = sinh (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = sin (A) ;
    C2 = sin (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = tanh (A) ;
    C2 = tanh (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = tan (A) ;
    C2 = tan (G) ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

    C1 = A.' ;
    C2 = G.' ;
    err = norm (C1-C2, 1) ;
    assert (err == 0) ;

    C1 = A' ;
    C2 = G' ;
    err = norm (C1-C2, 1) ;
    assert (err == 0) ;

    if (isfloat (G))
        C1 = eps (abs (A)) ;
        C2 = eps (G) ;
        err = norm (C1-C2, 1) ;
        assert (err < tol) ;
    end

    C1 = A+0 ;
    C2 = G+0 ;
    err = norm (C1-C2, 1) ;
    assert (err == 0) ;

    C1 = 0+A ;
    C2 = 0+G ;
    err = norm (C1-C2, 1) ;
    assert (err == 0) ;

    I = GrB ([1 2 3]) ;
    J = GrB ([3 1 2]) ;

    if (have_octave)
        % Octave does not allow indexing built-in matrices with objects.
        C1 = A (double (I), double (J)) ;
        C2 = G (double (I), double (J)) ;
    else
        % MATLAB sees the GrB/double method and does the typecasting itself.
        C1 = A (I,J) ;
        C2 = G (I,J) ;
    end
    err = norm (C1-C2, 1) ;
    assert (err == 0) ;

    C1 = A.^1 ;
    C2 = G.^1 ;
    err = norm (C1-C2, 1) ;
    assert (err < tol) ;

end


