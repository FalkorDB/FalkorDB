function test81
%TEST81 test GrB_Matrix_extract with index range, stride, & backwards

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test81:  GrB_Matrix_extract with index range, stride, backwards\n') ;

rng ('default') ;

n = 10 ;
A = sprand (n, n, 0.5) ;
S = sparse (n, n) ;

Ahyper.matrix = A ;
Ahyper.is_hyper = true ;
Ahyper.is_csc = true ;

for jlen = 1:2:n
    JJ {jlen} = randperm (n) ;
end

for ilo = 1:2 % 1:2:n
    for ihi = [4 8] % 1:2:n
        for i_inc = [-10 2 inf] % [-n:n inf]
            clear I
            I.begin = ilo-1 ;
            I.end = ihi-1 ;
            if (isfinite (i_inc))
                I.inc = i_inc ;
                iinc = i_inc ;
            else
                iinc = 1 ;
            end
            clear Ivec
            I_vec = [I.begin, I.end, iinc]' ;
            Ivec.sparsity = 8 ;         % full
            if (nnz (I_vec) == 3)
                I_vec = sparse (I_vec) ;
                Ivec.sparsity = 2 ;     % sparse
            end
            Ivec.matrix = I_vec ;
            if (ihi == 8)
                Ivec.sparsity = 4 ;     % bitmap
            end
            Ivec.class = 'double' ;

            for jlen = [1:2:n]
                clear J
                J = JJ {jlen} ;
                J = J (1:jlen) ;
                J0 = (uint64 (J) - 1)' ;
                C1 = A (ilo:iinc:ihi, J) ;
                [sm sn] = size (C1) ;
                S = sparse (sm, sn) ;
                C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I, J0, [ ]) ;
                assert (isequal (C1, C2.matrix)) ;
                C3 = GB_mex_Matrix_extract (S, [ ], [ ], ...
                    Ahyper, I, J0, [ ]) ;
                assert (isequal (C1, C3.matrix)) ;

                clear desc
                desc.rowindex_list = 'is_stride' ;
                J0_double = double (J0) ;
                C4 = GB_mex_Matrix_extract (S, [ ], [ ], A, Ivec, J0_double, desc, 1) ;
                assert (isequal (C1, C4.matrix)) ;

                clear Jvec
                C6 = A (ilo:iinc:ihi, 1:4) ;
                Jvec.matrix = [2 3 4 1]' ;
                Jvec.sparsity = 8 ;
                Jvec.class = 'double' ;
                desc.colindex_list = 'use_indices' ;
%               Jvec
%               Jvec.matrix
                S = sparse (sm, 4) ;
                C5 = GB_mex_Matrix_extract (S, [ ], [ ], A, Ivec, Jvec, desc, 1) ;
%               C6
%               C5.matrix
                assert (isequal (C6, C5.matrix)) ;

                Iv = [1 2]' ;
                try
                    C4 = GB_mex_Matrix_extract (S, [ ], [ ], A, Iv, J0, ...
                        desc, 1) ;
                    ok = 0 ;
                catch me
                    ok = 1 ;
                end
                assert (ok) ;

            end

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
                        Jvec = [J.begin, J.end, jinc]' ;

                        C1 = A (ilo:iinc:ihi, jlo:jinc:jhi) ;
                        [sm sn] = size (C1) ;
                        S = sparse (sm, sn) ;

                        C2 = GB_mex_Matrix_extract (S, [ ], [ ], A, I, J, [ ]) ;
                        assert (isequal (C1, C2.matrix)) ;

                        C3 = GB_mex_Matrix_extract (S, [ ], [ ], ...
                            Ahyper, I, J, [ ]) ;
                        assert (isequal (C1, C3.matrix)) ;

                        clear desc
                        desc.rowindex_list = 'is_stride' ;
                        desc.colindex_list = 'is_stride' ;
                        C4 = GB_mex_Matrix_extract (S, [ ], [ ], A, ...
                            Ivec, Jvec, desc, 1) ;
                        assert (isequal (C1, C4.matrix)) ;

                    end
                end
            end
        end
    end
end

fprintf ('\ntest81: all tests passed\n') ;

