function gbtest80
%GBTEST80 test complex division and power
% Tests all real, inf, and nan cases.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

list = [-2:0.5:2 inf -inf nan] ;

fprintf ('gbtest80: test complex division and power') ;

maxerr = 0 ;
maxerr_single = 0 ;
stol = 1e-5 ;
dtol = 1e-14 ;

for xr = list
    fprintf ('\n%g ', xr) ;
    % fprintf ('\n------------------------------------------\n') ;
    for xi = list
        fprintf ('.')
        % fprintf ('\n') ;
        X = complex (xr, xi) ;
        GX = GrB (X) ;
        Xs = single (X) ;
        GXs = GrB (Xs) ;

        for yr = list
            for yi = list

                Y = complex (yr, yi) ;
                GY = GrB (Y) ;
                Ys = single (Y) ;
                GYs = GrB (Ys) ;

                Z = X / Y ;
                Z2 = GX / GY ;
                [err, errnan] = gbtest_err (Z, Z2) ;
                znorm = abs (Z) ;
                if (znorm > 1)
                    err = err / znorm ;
                end
                if (~errnan)
                    maxerr = max (maxerr, err) ;
                end
                if (err > dtol)
                    fprintf ('(%g,%g) / (%g,%g) = (%g,%g) (%g,%g)', ...
                        xr, xi, yr, yi, real (Z), imag (Z), ...
                        real (Z2), imag (Z2)) ;
                    fprintf (' err: %g', err) ;
                    fprintf (' DOUBLE DIFFERS') ;
                    fprintf ('\n') ;
                    % pause
                end

                Z = Xs / Ys ;
                Z2 = GXs / GYs ;
                [err, errnan] = gbtest_err (Z, Z2) ;
                znorm = abs (Z) ;
                if (znorm > 1)
                    err = err / znorm ;
                end
                if (~errnan)
                    maxerr_single = max (maxerr_single, err) ;
                end
                if (err > stol)
                    fprintf ('(%g,%g) / (%g,%g) = (%g,%g) (%g,%g)', ...
                        xr, xi, yr, yi, real (Z), imag (Z), ...
                        real (Z2), imag (Z2)) ;
                    fprintf (' err: %g', err) ;
                    fprintf (' SINGLE DIFFERS') ;
                    fprintf ('\n') ;
                    % pause
                end

                Z = X .^ Y ;
                Z2 = GX .^ GY ;
                [err, errnan] = gbtest_err (Z, Z2) ;
                znorm = abs (Z) ;
                if (znorm > 1)
                    err = err / znorm ;
                end
                if (~errnan)
                    maxerr = max (maxerr, err) ;
                end
                if (err > dtol)
                    fprintf ('(%g,%g) .^ (%g,%g) = (%g,%g) (%g,%g)', ...
                        xr, xi, yr, yi, real (Z), imag (Z), ...
                        real (Z2), imag (Z2)) ;
                    fprintf (' err: %g', err) ;
                    fprintf (' DOUBLE DIFFERS') ;
                    fprintf ('\n') ;
                    % pause
                end

                Z = Xs .^ Ys ;
                Z2 = GXs .^ GYs ;
                [err, errnan] = gbtest_err (Z, Z2) ;
                znorm = abs (Z) ;
                if (znorm > 1)
                    err = err / znorm ;
                end
                if (~errnan)
                    maxerr_single = max (maxerr_single, err) ;
                end
                if (err > stol)
                    fprintf ('(%g,%g) .^ (%g,%g) = (%g,%g) (%g,%g)', ...
                        xr, xi, yr, yi, real (Z), imag (Z), ...
                        real (Z2), imag (Z2)) ;
                    fprintf (' err: %g', err) ;
                    fprintf (' SINGLE DIFFERS') ;
                    fprintf ('\n') ;
                    % pause
                end

            end
        end
    end
end

fprintf ('\nmaxerr: %g %g\n', maxerr, maxerr_single) ;
assert (maxerr < dtol)
assert (maxerr_single < stol)

fprintf ('\ngbtest80: all tests passed\n') ;

