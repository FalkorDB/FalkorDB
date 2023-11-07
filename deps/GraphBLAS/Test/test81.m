function test81
%TEST81 test GrB_Matrix_extract with index range, stride, & backwards

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test81:  GrB_Matrix_extract with index range, stride, backwards\n') ;

rng ('default') ;

n = 10 ;
A = sprand (n, n, 0.5) ;
S = sparse (n, n) ;

Ahyper.matrix = A ;
Ahyper.is_hyper = true ;
Ahyper.is_csc = true ;

for ilo = 1:2:n
    for ihi = 1:2:n
        fprintf ('#') ;
        for i_inc = [-n:n inf]
            clear I
            I.begin = ilo-1 ;
            I.end = ihi-1 ;
            if (isfinite (i_inc))
                I.inc = i_inc ;
                iinc = i_inc ;
            else
                iinc = 1 ;
            end

            fprintf (':') ;
            for jlen = [1:2:n]
                clear J
                J = randperm (n) ;
                J = J (1:jlen) ;
                J0 = uint64 (J) - 1 ;
                C1 = A (ilo:iinc:ihi, J) ;
                [sm sn] = size (C1) ;
                S = sparse (sm, sn) ;
                C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I, J0, [ ]) ;
                assert (isequal (C1, C2.matrix)) ;
                C3 = GB_mex_Matrix_extract (S, [ ], [ ], ...
                    Ahyper, I, J0, [ ]) ;
                assert (isequal (C1, C3.matrix)) ;
            end

            fprintf ('.') ;
            for jlo = 1:2:n
                for jhi = 1:2:n
                    for j_inc = [-n:n inf]

                        clear J
                        J.begin = jlo-1 ;
                        J.end = jhi-1 ;
                        if (isfinite (j_inc))
                            J.inc = j_inc ;
                            jinc = j_inc ;
                        else
                            jinc = 1 ;
                        end

                        C1 = A (ilo:iinc:ihi, jlo:jinc:jhi) ;
                        [sm sn] = size (C1) ;
                        S = sparse (sm, sn) ;

                        C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I, J, [ ]) ;
                        assert (isequal (C1, C2.matrix)) ;

                        C3 = GB_mex_Matrix_extract (S, [ ], [ ], ...
                            Ahyper, I, J, [ ]) ;
                        assert (isequal (C1, C3.matrix)) ;

                    end
                end
            end

        end
    end
end

fprintf ('\ntest81: all tests passed\n') ;

