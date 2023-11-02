function test54
%TEST54 test GB_subref: numeric case with I=lo:hi, J=lo:hi

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest54: ==== quick test for subref and assign (lo:stride:hi):\n') ;
clear

m = 5 ;
n = 7 ;
A = sprandn (m,n,0.4) ;

for ilo = 1:m
    for ihi = ilo:m
        for jlo = 1:n
            for jhi = jlo:m

                C = A (ilo:ihi,jlo:jhi) ;

                I1 = uint64 (ilo:ihi) - 1 ;
                J1 = uint64 (jlo:jhi) - 1 ;
                C1 = GB_mex_Matrix_subref (A, I1, J1) ;
                assert (isequal (C, C1)) ;

                I2.begin = ilo-1 ;
                I2.end   = ihi-1 ;
                J2.begin = jlo-1 ;
                J2.end   = jhi-1 ;
                C2 = GB_mex_Matrix_subref (A, I2, J2) ;
                assert (isequal (C, C2)) ;

            end
        end
    end
end

C = sprandn (m, n, 0.4) ;

for ilo = 1:m
    for ihi = ilo:m

        sm = ihi-ilo+1 ;
        S = sprandn (sm, n, 0.4) ;

        C0 = C ;
        C0 (ilo:ihi,:) = S ;

        I1 = uint64 (ilo:ihi) - 1 ;
        C1 = GB_mex_subassign (C, [ ], [ ], S, I1, [ ], [ ]) ;
        assert (isequal (C0, C1.matrix)) ;

        C1b = GB_mex_assign (C, [ ], [ ], S, I1, [ ], [ ]) ;
        assert (isequal (C0, C1b.matrix)) ;

        I2.begin = ilo-1 ;
        I2.end   = ihi-1 ;
        C2 = GB_mex_subassign (C, [ ], [ ], S, I2, [ ], [ ]) ;
        assert (isequal (C0, C2.matrix)) ;

        C2b = GB_mex_assign (C, [ ], [ ], S, I2, [ ], [ ]) ;
        assert (isequal (C0, C2b.matrix)) ;

        S = sprandn (m, sm, 0.4) ;

        C0 = C ;
        C0 (:,ilo:ihi) = S ;

        C1 = GB_mex_subassign (C, [ ], [ ], S, [ ], I1, [ ]) ;
        assert (isequal (C0, C1.matrix)) ;

        C1b = GB_mex_assign (C, [ ], [ ], S, [ ], I1, [ ]) ;
        assert (isequal (C0, C1b.matrix)) ;

        C2 = GB_mex_subassign (C, [ ], [ ], S, [ ], I2, [ ]) ;
        assert (isequal (C0, C2.matrix)) ;

        C2b = GB_mex_assign (C, [ ], [ ], S, [ ], I2, [ ]) ;
        assert (isequal (C0, C2b.matrix)) ;

        for jlo = 1:n
            for jhi = jlo:n

                sm = ihi-ilo+1 ;
                sn = jhi-jlo+1 ;
                S = sprandn (sm, sn, 0.4) ;

                C0 = C ;
                C0 (ilo:ihi,jlo:jhi) = S ;

                I1 = uint64 (ilo:ihi) - 1 ;
                J1 = uint64 (jlo:jhi) - 1 ;
                C1 = GB_mex_subassign (C, [ ], [ ], S, I1, J1, [ ]) ;
                assert (isequal (C0, C1.matrix)) ;

                C1b = GB_mex_assign (C, [ ], [ ], S, I1, J1, [ ]) ;
                assert (isequal (C0, C1b.matrix)) ;

                I2.begin = ilo-1 ;
                I2.end   = ihi-1 ;
                J2.begin = jlo-1 ;
                J2.end   = jhi-1 ;
                C2 = GB_mex_subassign (C, [ ], [ ], S, I2, J2, [ ]) ;
                assert (isequal (C0, C2.matrix)) ;

                C2b = GB_mex_assign (C, [ ], [ ], S, I2, J2, [ ]) ;
                assert (isequal (C0, C2b.matrix)) ;

            end
        end
    end
end

clear C0 C1 C1b C2 C2b I1 J1 I2 J2

fprintf ('longer tests: ') ;
jinc_list = unique ([-1 1 3]) ;

for ibegin = 1:m
    for iend = 1:m
        for iinc = -m:m

            I = ibegin:iinc:iend ;
            I1.begin = ibegin - 1 ;
            I1.inc = iinc ;
            I1.end = iend - 1 ;

            fprintf ('.') ;

            for jbegin = 1:3:n
                for jend = 1:3:n
                    for jinc = jinc_list

                        J = jbegin:jinc:jend ;
                        J1.begin = jbegin - 1 ;
                        J1.inc = jinc ;
                        J1.end = jend - 1 ;

                        sm = length (I) ;
                        sn = length (J) ;
                        S = sprandn (sm, sn, 0.4) ;

                        C0 = C ;
                        C0 (I,J) = S ;

                        % fprintf ('\nsubassign:\n') ;
                        C1 = GB_mex_subassign (C, [ ], [ ], S, I1, J1, [ ]) ;
                        assert (isequal (C0, C1.matrix)) ;

                        % fprintf ('\nassign:\n') ;
                        C2 = GB_mex_assign (C, [ ], [ ], S, I1, J1, [ ]) ;
                        assert (isequal (C0, C2.matrix)) ;

                        R = C(I,J) ;
                        R1 = GB_mex_Matrix_subref (C, I1, J1) ;
                        assert (isequal (R, R1)) ;

                    end
                end
            end
        end
    end
end


fprintf ('\ntest54: all tests passed\n') ;
