function test115
%TEST115 test GB_assign, scalar expansion and zombies, with duplicates

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

    n = 200 ;
    C = sparse (rand (n)) ;
    C (:,1) = sparse (rand (n,1) > 0.5) ;

    ntrials = 10 ;

    for k = 1:2:ntrials

        % make some zombies
        Work (k).A = sprandn (n, n, 0.1) ;
        Work (k).I = [ ] ;
        Work (k).J = [ ] ;
        Work (k).Mask = sparse (ones (n)) ;
        Work (k).accum = [ ] ;
        Work (k).desc = [ ] ;
        Work (k).scalar = 0 ;

        % scalar expansion
        if (rand (1) > 0.5)
            % use non-opaque scalar
            Work (k+1).A = sparse (pi) ;
            use_GrB_Scalar = 1 ;
        else
            % use GrB_Scalar
            Work (k+1).A = sparse (pi * (rand(1) > 0.5)) ;
            use_GrB_Scalar = 2 ;
        end
        Work (k+1).I = uint64 (irand (1, n, n, 1) - 1) ;
        Work (k+1).J = uint64 (irand (1, n, n, 1) - 1) ;
        Work (k+1).Mask = sparse (ones (n)) ;
        Work (k+1).accum = [ ] ;
        Work (k+1).desc = [ ] ;
        Work (k+1).scalar = use_GrB_Scalar ;
    end

    C1 = GB_mex_assign  (C, Work) ;     % WORK_ASSIGN

    C2 = C ;
    for k = 1:ntrials
        M = Work (k).Mask ;
        A = Work (k).A ;
        I = double (Work (k).I + 1) ;
        J = double (Work (k).J + 1) ;
        scalar = Work (k).scalar ;
        C2 = GB_spec_subassign (C2, M, [ ], A, I, J, [ ], scalar) ;
    end

    GB_spec_compare (C1, C2) ;

    for C_sparsity = [2 4]
        for M_sparsity = [2 4]
            ctrl = [C_sparsity M_sparsity] ;
            C1 = GB_mex_assign (C, Work, ctrl) ; % WORK_ASSIGN
            GB_spec_compare (C1, C2) ;
        end
    end

fprintf ('\ntest115: all tests passed\n') ;

