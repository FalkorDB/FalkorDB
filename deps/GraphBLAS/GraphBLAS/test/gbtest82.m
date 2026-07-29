function gbtest82 (ghb)
%GBTEST82 test complex A*B, A'*B, A*B', A'*B', A+B

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

nlist = [1 4 10] ;
r = complex ([-1 1]) ;
maxerr = 0 ;
for m = nlist
    for n = nlist
        for k = nlist
            A = gtb_random (ghb, k, m, (m*5)/(k*m), 'range', r) ;
            B = gtb_random (ghb, k, n, (n*5)/(k*n), 'range', r) ;
            C = double (A).'*double (B) ;
            C2 = gtb_mxm (ghb, A, '+.*', B, struct ('in0', 'transpose')) ;
            err = norm (C-C2,1) ;
            maxerr = max (maxerr, err) ;
            assert (err < 1e-12)
        end
    end
end
fprintf ('All complex A''*B tests passed, maxerr %g\n', maxerr) ;

maxerr = 0 ;
for m = nlist
    for n = nlist

            A = gtb_random (ghb, m, n, (m*5)/(k*m), 'range', r) ;
            B = gtb_random (ghb, m, n, (n*5)/(k*n), 'range', r) ;
            C = double (A) + double (B) ;
            C2 = A + B  ;
            err = norm (C-C2,1) ;
            maxerr = max (maxerr, err) ;
            assert (err < 1e-12)
    end
end
fprintf ('All complex A+B tests passed, maxerr %g\n', maxerr) ;

maxerr = 0 ;
for m = nlist
    for n = nlist
        for k = nlist
            for at = 0:1
                for bt = 0:1
                    if (at)
                        A = gtb_random (ghb, k, m, (n*5)/(k*m), 'range', r) ;
                    else
                        A = gtb_random (ghb, m, k, (m*5)/(k*m), 'range', r) ;
                    end
                    if (bt)
                        B = gtb_random (ghb, n, k, (m*5)/(k*m), 'range', r) ;
                    else
                        B = gtb_random (ghb, k, n, (m*5)/(k*m), 'range', r) ;
                    end

                    desc = struct ;
                    if (at)
                        desc.in0 = 'transpose' ;
                    end
                    if (bt)
                        desc.in1 = 'transpose' ;
                    end

                    M = sparse (m, n) ;
                    M (1,1) = 1 ; %#ok

                    C = gtb_mxm (ghb, A, '+.*', B, desc) ;
                    Cin = gtb (ghb, m, n, 'double complex') ;
                    CM = gtb_mxm (ghb, Cin, M, A, '+.*', B, desc) ;

                    A = double (A) ;
                    B = double (B) ;

                    if (at)
                        if (bt)
                            C2 = A.'*B.'  ;
                            CM2 = (A.'*B.') .* M ;
                        else
                            C2 = A.'*B  ;
                            CM2 = (A.'*B) .* M ;
                        end
                    else
                        if (bt)
                            C2 = A*B.'  ;
                            CM2 = (A*B.') .* M ;
                        else
                            C2 = A*B  ;
                            CM2 = (A*B) .* M ;
                        end
                    end
                    err = norm (C-C2,1) ;
                    maxerr = max (maxerr, err) ;
                    assert (err < 1e-12)

                    err = norm (CM-CM2,1) ;
                    maxerr = max (maxerr, err) ;
                    assert (err < 1e-12)

                end
            end
        end
    end
end

fprintf ('maxerr: %g\n', maxerr) ;
fprintf ('gbtest82 (%d): all tests passed\n', ghb) ;

