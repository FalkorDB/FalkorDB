function gbtest79
%GBTEST79 test real power
% Tests all real, inf, and nan cases.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

list = [-2:0.5:2 inf -inf nan] ;

fprintf ('gbtest79: test power') ;

maxerr = 0 ;
maxerr_single = 0 ;

for xr = list
    fprintf ('.') ;

    X = xr ;
    GX = GrB (X) ;
    Xs = single (X) ;
    GXs = GrB (Xs) ;

    for yr = list
        Y = yr ;
        GY = GrB (Y) ;
        Ys = single (Y) ;
        GYs = GrB (Ys) ;

        Z = X .^ Y ;
        Z2 = GX .^ GY ;
        [err, errnan] = gbtest_err (Z, Z2) ;
        znorm = abs (Z) ;
        if (znorm > 0)
            err = err / znorm ;
        end

        if (err > 1e-14)
            fprintf ('(%g) .^ (%g) = (%g,%g) (%g,%g)', ...
                xr, yr, real (Z), imag (Z), ...
                real (Z2), imag (Z2)) ;
            fprintf (' err: %g', err) ;
            fprintf (' DOUBLE DIFFERS') ;
            fprintf ('\n') ;
            % pause
        end
        if (~errnan)
            maxerr = max (maxerr, err) ;
        end
        % assert (err < 1e-14)

        Z = Xs .^ Ys ;
        Z2 = GXs .^ GYs ;
        [err, errnan] = gbtest_err (Z, Z2) ;
        znorm = abs (Z) ;
        if (znorm > 0)
            err = err / znorm ;
        end
        if (~errnan)
            maxerr_single = max (maxerr_single, err) ;
        end

        if (err > 1e-6)
            fprintf ('(%g) .^ (%g) = (%g,%g) (%g,%g)', ...
                xr, yr, real (Z), imag (Z), ...
                real (Z2), imag (Z2)) ;
            fprintf (' err: %g', err) ;
            fprintf (' SINGLE DIFFERS') ;
            fprintf ('\n') ;
            % pause
        end

        % assert (maxerr_single < 1e-6) ;
    end
end

A = int32 (magic (4)) ;
B = int32 (2 * rand (4)) ;
GA = GrB (A) ;
GB = GrB (B) ;
C1 = A.^B ;
C2 = GA.^GB ;
assert (isequal (C1, C2)) ;

fprintf ('\nmaxerr: %g %g\n', maxerr, maxerr_single) ;
assert (maxerr < 1e-14)
assert (maxerr_single < 1e-6)

fprintf ('\ngbtest79: all tests passed\n') ;

