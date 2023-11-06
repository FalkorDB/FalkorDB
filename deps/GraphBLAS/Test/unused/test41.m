function test41
%TEST41 test AxB

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n -------------- simple GB_mex_AxB numeric tests\n') ;

rng ('default') ;

for at = [false true]
    for bt = [false true]
        for ct = [false true]


                % create the problem

                % A or A' will be 4-by-5
                if (at)
                    A = sprand (5000, 4000, 0.01) ;
                else
                    A = sprand (4000, 5000, 0.01) ;
                end

                % B or B' will be 5-by-3
                if (bt)
                    B = sprand (3000, 5000, 0.01) ;
                else
                    B = sprand (5000, 3000, 0.01) ;
                end

                % C will be 4-by-3
                % C' will be 3-by-4

                fprintf ('\nat %d bt %d ct %d\n', at, bt, ct) ;

                fprintf ('built-in:  ') ;
                tic
                if (at)
                    if (bt)
                        if (ct)
                            C = (A'*B')' ;
                        else
                            C = (A'*B') ;
                        end
                    else
                        if (ct)
                            C = (A'*B)' ;
                        else
                            C = (A'*B) ;
                        end
                    end
                else
                    if (bt)
                        if (ct)
                            C = (A*B')' ;
                        else
                            C = (A*B') ;
                        end
                    else
                        if (ct)
                            C = (A*B)' ;
                        else
                            C = (A*B) ;
                        end
                    end
                end
                toc

                fprintf ('GrB num: ') ;
                tic
                S = GB_mex_AxB (A, B, at, bt) ;
                if (ct)
                    S = S' ;
                end
                toc
                assert (isequal (S, C)) ;

        end
    end
end

fprintf ('\ntest41: all tests passed\n') ;

