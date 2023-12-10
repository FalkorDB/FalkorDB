function gbtest82
%GBTEST82 test complex A*B, A'*B, A*B', A'*B', A+B

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

nlist = [1 4 10] ;
r = complex ([-1 1]) ;
maxerr = 0 ;
for m = nlist
    for n = nlist
        for k = nlist
            A = GrB.random (k, m, (m*5)/(k*m), 'range', r) ;
            B = GrB.random (k, n, (n*5)/(k*n), 'range', r) ;
            C = double (A).'*double (B) ;
            C2 = GrB.mxm (A, '+.*', B, struct ('in0', 'transpose')) ;
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

            A = GrB.random (m, n, (m*5)/(k*m), 'range', r) ;
            B = GrB.random (m, n, (n*5)/(k*n), 'range', r) ;
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
                        A = GrB.random (k, m, (n*5)/(k*m), 'range', r) ;
                    else
                        A = GrB.random (m, k, (m*5)/(k*m), 'range', r) ;
                    end
                    if (bt)
                        B = GrB.random (n, k, (m*5)/(k*m), 'range', r) ;
                    else
                        B = GrB.random (k, n, (m*5)/(k*m), 'range', r) ;
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

                    C = GrB.mxm (A, '+.*', B, desc) ;
                    Cin = GrB (m, n, 'double complex') ;
                    CM = GrB.mxm (Cin, M, A, '+.*', B, desc) ;

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
fprintf ('gbtest82: all tests passed\n') ;

