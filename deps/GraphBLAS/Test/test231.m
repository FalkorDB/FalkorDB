function test231
%TEST231 test GrB_select with idxunp

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, ~, idxunops] = GB_spec_opsall ;
ops = idxunops ;
types_all = types.all ;

fprintf ('\n--- testing select with idxunops\n') ;
rng ('default') ;

ds.inp0 = 'tran' ;

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

% create test matrices
m = 4 ;
n = 5 ;
G {1} = sprand (m, n, 0.8) ;
S {1} = sprand (m, n, 0.5) ;
G {2} = sprand (m, 1, 0.8) ;
S {2} = sprand (m, 1, 0.5) ;
hi = 5 ;
lo = -1 ;

n_operators = 0 ;
for k2 = 1:length(ops)
    opname = ops {k2} ;

    % fprintf ('\n%-10s ', opname) ;

    if (test_contains (opname, 'value'))
        types = types_all ;
        ythunks = 1 ;
        sparsities = 1 ;
        cscs = 1 ;
    else
        types = {'int64'} ;
        ythunks = [-3 0 1] ;
        sparsities = [1 4] ;
        cscs = [0 1] ;
    end

    if (isequal (opname, 'rowindex'))
        nmat = 2 ;
    else
        nmat = 1 ;
    end

    for k1 = 1:length (types)
        type = types {k1} ;

        % create the op
        clear op
        op.opname = opname ;
        op.optype = type ;
        [is_idxunop, ztype] = GB_spec_is_idxunop (opname, type) ;
        if (~is_idxunop)
            continue ;
        end

        for kmat = 1:nmat

            % create the test matrix
            Amat = (hi*G{kmat}-lo) .* S{kmat} ;
            [m n] = size (Amat) ;
            Cmat = sparse (m, n) ;

            C.matrix = Cmat ;
            C.class = ztype ;

            CT.matrix = Cmat' ;
            CT.class = ztype ;

            A.matrix = Amat ;
            A.class = type ;

            B.matrix = spones (Amat) ;
            B.class = type ;
            B.iso = true ;

            for ythunk = ythunks
                y.matrix = ythunk ;
                y.class = type ;

                for csc = cscs
                    A.is_csc = csc ;
                    C.is_csc = csc ;
                    CT.is_csc = csc ;

                    for sparsity = sparsities % [1 4]
                        A.sparsity = sparsity ;

                        C1 = GB_mex_select_idxunop  (C, [],[], op,0, A, y, []) ;
                        C2 = GB_spec_select_idxunop (C, [],[], op,   A, y, []) ;
                        GB_spec_compare (C1, C2) ;

                        C1 = GB_mex_select_idxunop  (C, [],[], op,0, B, y, []) ;
                        C2 = GB_spec_select_idxunop (C, [],[], op,   B, y, []) ;
                        GB_spec_compare (C1, C2) ;

                        C1 = GB_mex_select_idxunop  (CT,[],[], op,0, A, y, ds) ;
                        C2 = GB_spec_select_idxunop (CT,[],[], op,   A, y, ds) ;
                        GB_spec_compare (C1, C2) ;

                        if (track_coverage)
                            c = sum (GraphBLAS_grbcov > 0) ;
                            d = c - clast ;
                            if (d > 0)
                                fprintf (...
                                '[%10s, %15s, %2d,%2d,%2d,%2d]', ...
                                opname, type, kmat, ythunk, ...
                                csc, sparsity) ;
                                fprintf (' (%d, %d)\n', d, c - cfirst) ;
                            end
                            clast = c ;
                        else
                            fprintf ('.') ;
                        end

                    end
                end
            end
        end
    end
end

fprintf ('\ntest231: all tests passed\n') ;

