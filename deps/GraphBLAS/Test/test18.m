function test18
%TEST18 test GrB_eWiseAdd, GxB_eWiseUnion, and GrB_eWiseMult

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
bin_ops = binops.all ;
types = types.all ;

fprintf ('test18 ------quick tests of GrB_eWiseAdd, eWiseUnion, and eWiseMult\n') ;

rng ('default') ;
m = [10] ; 
n = [10] ; 

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
end

for k1 = [1 2 11 12 13]
    type = types {k1}  ;

    for k2 = 1:length(bin_ops)
        binop = bin_ops {k2}  ;
        op.opname = binop ;
        op.optype = type ;
        try
            GB_spec_operator (op) ;
        catch
            continue
        end

        if (test_contains (type, 'single'))
            tol = 1e-5 ;
        elseif (test_contains (type, 'double'))
            tol = 1e-12 ;
        else
            tol = 0 ;
        end

        % avoid creating nans when testing pow
        if (isequal (binop, 'pow'))
            scale = 2 ;
        else
            scale = 100 ;
        end

        Amat = sparse (scale * sprandn (m,n, 0.2)) ;
        Bmat = sparse (scale * sprandn (m,n, 0.2)) ;
        Cmat = sparse (scale * sprandn (m,n, 0.2)) ;
        w    = sparse (scale * sprandn (m,1, 0.2)) ;
        uvec = sparse (scale * sprandn (m,1, 0.2)) ;
        vvec = sparse (scale * sprandn (m,1, 0.2)) ;

        % these tests do not convert real A and B into complex C
        % when C = A.^B.  So ensure the test matrices are all positive.
        if (isequal (binop, 'pow'))
            Amat = abs (Amat) ;
            Bmat = abs (Bmat) ;
            Cmat = abs (Cmat) ;
            w    = abs (w) ;
            uvec = abs (uvec) ;
            vvec = abs (vvec) ;
        end

        Maskmat = sprandn (m,n,0.2) ~= 0 ;
        maskvec = sprandn (m,1,0.2) ~= 0 ;

        accum = ''  ;
        ntypes = 1 ;
        % k5 is unused, but required to get the right random
        % test matrices above.
        k5 = randi ([1 ntypes]) ;

        % skip all but these operators:
        if (((isequal (binop, 'first')) && isequal (type, 'logical')) || ...
            ((isequal (binop, 'second')) && isequal (type, 'logical')) || ...
            ((isequal (binop, 'pair')) && isequal (type, 'logical')) || ...
            ((isequal (binop, 'plus')) && isequal (type, 'double')))
        else
            continue
        end

        fprintf ('\n[ %s %s ] ', binop, type) ;

        % create a very sparse matrix mask
        Maskmat2 = sparse (m,n) ;
        T = Amat .* Bmat ;
        [i j x] = find (T) ;
        if (length (i) > 0)
            Maskmat2 (i(1), j(1)) = 1 ;
        end
        T = (Amat ~= 0) & (Bmat == 0) ;
        [i j x] = find (T) ;
        if (length (i) > 0)
            Maskmat2 (i(1), j(1)) = 1 ;
        end
        T = (Amat == 0) & (Bmat ~= 0) ;
        [i j x] = find (T) ;
        if (length (i) > 0)
            Maskmat2 (i(1), j(1)) = 1 ;
        end
        clear T i j x

        % create a very sparse vector mask
        maskvec2 = sparse (m,1) ;
        T = uvec .* vvec ;
        [i j x] = find (T) ;
        if (length (i) > 0)
            maskvec2 (i(1), j(1)) = 1 ;
        end
        T = (uvec ~= 0) & (vvec == 0) ;
        [i j x] = find (T) ;
        if (length (i) > 0)
            maskvec2 (i(1), j(1)) = 1 ;
        end
        T = (uvec == 0) & (vvec ~= 0) ;
        [i j x] = find (T) ;
        if (length (i) > 0)
            maskvec2 (i(1), j(1)) = 1 ;
        end
        clear T i j x

        ATmat = Amat' ;
        BTmat = Bmat' ;

        dnn.mask = 'default' ;
        dtn.mask = 'default' ;
        dnt.mask = 'default' ;
        dtt.mask = 'default' ;

        dnn.outp = 'default' ;
        dtn.outp = 'default' ;
        dnt.outp = 'default' ;
        dtt.outp = 'default' ;

        for A_is_hyper = 0:1
        for A_is_csc   = 0:1
        for B_is_hyper = 0:1
        for B_is_csc   = 0:1

            clear A
            A.matrix = Amat ;
            A.is_hyper = A_is_hyper ;
            A.is_csc   = A_is_csc   ;
            A.class = op.optype ;

            clear AT
            AT.matrix = ATmat ;
            AT.is_hyper = A_is_hyper ;
            AT.is_csc   = A_is_csc   ;
            AT.class = op.optype ;

            clear B
            B.matrix = Bmat ;
            B.is_hyper = B_is_hyper ;
            B.is_csc   = B_is_csc   ;
            B.class = op.optype ;

            clear BT
            BT.matrix = BTmat ;
            BT.is_hyper = B_is_hyper ;
            BT.is_csc   = B_is_csc   ;
            BT.class = op.optype ;

            clear C
            C.matrix = Cmat ;
            C.is_hyper = 0 ;
            C.is_csc   = 0   ;

            clear u
            u.matrix = uvec ;
            u.is_csc = true ;
            u.class = op.optype ;

            clear v
            v.matrix = vvec ;
            v.is_csc = true ;
            v.class = op.optype ;

            %---------------------------------------
            % A+B
            %---------------------------------------

            C0 = GB_spec_Matrix_eWiseAdd ...
                (C, [ ], accum, op, A, B, dnn);
            C1 = GB_mex_Matrix_eWiseAdd ...
                (C, [ ], accum, op, A, B, dnn);
            GB_spec_compare (C0, C1, 0, tol) ;

            w0 = GB_spec_Vector_eWiseAdd ...
                (w, [ ], accum, op, u, v, dnn);
            w1 = GB_mex_Vector_eWiseAdd ...
                (w, [ ], accum, op, u, v, dnn);
            GB_spec_compare (w0, w1, 0, tol) ;

            %---------------------------------------
            % A+B with eWiseUnion
            %---------------------------------------

            C0 = GB_spec_Matrix_eWiseUnion ...
                (C, [ ], accum, op, A, 3, B, 2, dnn);
            C1 = GB_mex_Matrix_eWiseUnion ...
                (C, [ ], accum, op, A, 3, B, 2, dnn);
            GB_spec_compare (C0, C1, 0, tol) ;

            w0 = GB_spec_Vector_eWiseUnion ...
                (w, [ ], accum, op, u, 3, v, 2, dnn);
            w1 = GB_mex_Vector_eWiseUnion ...
                (w, [ ], accum, op, u, 3, v, 2, dnn);
            GB_spec_compare (w0, w1, 0, tol) ;

            %---------------------------------------
            % A'+B
            %---------------------------------------

            C0 = GB_spec_Matrix_eWiseAdd ...
                (C, [ ], accum, op, AT, B, dtn);
            C1 = GB_mex_Matrix_eWiseAdd ...
                (C, [ ], accum, op, AT, B, dtn);
            GB_spec_compare (C0, C1, 0, tol) ;

            %---------------------------------------
            % A+B'
            %---------------------------------------

            C0 = GB_spec_Matrix_eWiseAdd ...
                (C, [ ], accum, op, A, BT, dnt);
            C1 = GB_mex_Matrix_eWiseAdd ...
                (C, [ ], accum, op, A, BT, dnt);
            GB_spec_compare (C0, C1, 0, tol) ;

            %---------------------------------------
            % A'+B'
            %---------------------------------------

            C0 = GB_spec_Matrix_eWiseAdd ...
                (C, [ ], accum, op, AT, BT, dtt);
            C1 = GB_mex_Matrix_eWiseAdd ...
                (C, [ ], accum, op, AT, BT, dtt);
            GB_spec_compare (C0, C1, 0, tol) ;

            %---------------------------------------
            % A.*B
            %---------------------------------------

            C0 = GB_spec_Matrix_eWiseMult ...
                (C, [ ], accum, op, A, B, dnn);
            C1 = GB_mex_Matrix_eWiseMult ...
                (C, [ ], accum, op, A, B, dnn);
            GB_spec_compare (C0, C1, 0, tol) ;

            w0 = GB_spec_Vector_eWiseMult ...
                (w, [ ], accum, op, u, v, dnn);
            w1 = GB_mex_Vector_eWiseMult ...
                (w, [ ], accum, op, u, v, dnn);
            GB_spec_compare (w0, w1, 0, tol) ;

            %---------------------------------------
            % A'.*B
            %---------------------------------------

            C0 = GB_spec_Matrix_eWiseMult ...
                (C, [ ], accum, op, AT, B, dtn);
            C1 = GB_mex_Matrix_eWiseMult ...
                (C, [ ], accum, op, AT, B, dtn);
            GB_spec_compare (C0, C1, 0, tol) ;

            %---------------------------------------
            % A.*B'
            %---------------------------------------

            C0 = GB_spec_Matrix_eWiseMult ...
                (C, [ ], accum, op, A, BT, dnt);
            C1 = GB_mex_Matrix_eWiseMult ...
                (C, [ ], accum, op, A, BT, dnt);
            GB_spec_compare (C0, C1, 0, tol) ;

            %---------------------------------------
            % A'.*B'
            %---------------------------------------

            C0 = GB_spec_Matrix_eWiseMult ...
                (C, [ ], accum, op, AT, BT, dtt);
            C1 = GB_mex_Matrix_eWiseMult ...
                (C, [ ], accum, op, AT, BT, dtt);
            GB_spec_compare (C0, C1, 0, tol) ;

            %-----------------------------------------------
            % with mask
            %-----------------------------------------------

            for M_is_very_sparse = 0:1
            for M_is_hyper = 0:1
            for M_is_csc   = 0:1

                clear Mask mask
                if (M_is_very_sparse)
                    Mask.matrix = Maskmat2 ;
                    mask.matrix = maskvec2 ;
                else
                    Mask.matrix = Maskmat ;
                    mask.matrix = maskvec ;
                end
                Mask.is_hyper = M_is_hyper ;
                Mask.is_csc   = M_is_csc   ;
                mask.is_csc = true ;

                %---------------------------------------
                % A+B, with mask
                %---------------------------------------

                C0 = GB_spec_Matrix_eWiseAdd ...
                    (C, Mask, accum, op, A, B, dnn);
                C1 = GB_mex_Matrix_eWiseAdd ...
                    (C, Mask, accum, op, A, B, dnn);
                GB_spec_compare (C0, C1, 0, tol) ;

                w0 = GB_spec_Vector_eWiseAdd ...
                    (w, mask, accum, op, u, v, dnn);
                w1 = GB_mex_Vector_eWiseAdd ...
                    (w, mask, accum, op, u, v, dnn);
                GB_spec_compare (w0, w1, 0, tol) ;

                %---------------------------------------
                % A+B, with mask, eWiseUnion
                %---------------------------------------

                C0 = GB_spec_Matrix_eWiseUnion ...
                    (C, Mask, accum, op, A, 1, B, 2, dnn);
                C1 = GB_mex_Matrix_eWiseUnion ...
                    (C, Mask, accum, op, A, 1, B, 2, dnn);
                GB_spec_compare (C0, C1, 0, tol) ;

                w0 = GB_spec_Vector_eWiseUnion ...
                    (w, mask, accum, op, u, 1, v, 2, dnn);
                w1 = GB_mex_Vector_eWiseUnion ...
                    (w, mask, accum, op, u, 1, v, 2, dnn);
                GB_spec_compare (w0, w1, 0, tol) ;

                %---------------------------------------
                % A'+B, with mask
                %---------------------------------------

                C0 = GB_spec_Matrix_eWiseAdd ...
                    (C, Mask, accum, op, AT, B, dtn);
                C1 = GB_mex_Matrix_eWiseAdd ...
                    (C, Mask, accum, op, AT, B, dtn);
                GB_spec_compare (C0, C1, 0, tol) ;

                %---------------------------------------
                % A+B', with mask
                %---------------------------------------

                C0 = GB_spec_Matrix_eWiseAdd ...
                    (C, Mask, accum, op, A, BT, dnt);
                C1 = GB_mex_Matrix_eWiseAdd ...
                    (C, Mask, accum, op, A, BT, dnt);
                GB_spec_compare (C0, C1, 0, tol) ;

                %---------------------------------------
                % A'+B', with mask
                %---------------------------------------

                C0 = GB_spec_Matrix_eWiseAdd ...
                    (C, Mask, accum, op, AT, BT, dtt);
                C1 = GB_mex_Matrix_eWiseAdd ...
                    (C, Mask, accum, op, AT, BT, dtt);
                GB_spec_compare (C0, C1, 0, tol) ;

                %---------------------------------------
                % A.*B, with mask
                %---------------------------------------

                C0 = GB_spec_Matrix_eWiseMult ...
                    (C, Mask, accum, op, A, B, dnn);
                C1 = GB_mex_Matrix_eWiseMult ...
                    (C, Mask, accum, op, A, B, dnn);
                GB_spec_compare (C0, C1, 0, tol) ;

                w0 = GB_spec_Vector_eWiseMult ...
                    (w, mask, accum, op, u, v, dnn);
                w1 = GB_mex_Vector_eWiseMult ...
                    (w, mask, accum, op, u, v, dnn);
                GB_spec_compare (w0, w1, 0, tol) ;

                %---------------------------------------
                % A'.*B, with mask
                %---------------------------------------

                C0 = GB_spec_Matrix_eWiseMult ...
                    (C, Mask, accum, op, AT, B, dtn);
                C1 = GB_mex_Matrix_eWiseMult ...
                    (C, Mask, accum, op, AT, B, dtn);
                GB_spec_compare (C0, C1, 0, tol) ;

                %---------------------------------------
                % A.*B', with mask
                %---------------------------------------

                C0 = GB_spec_Matrix_eWiseMult ...
                    (C, Mask, accum, op, A, BT, dnt);
                C1 = GB_mex_Matrix_eWiseMult ...
                    (C, Mask, accum, op, A, BT, dnt);
                GB_spec_compare (C0, C1, 0, tol) ;

                %---------------------------------------
                % A'.*B', with mask
                %---------------------------------------

                C0 = GB_spec_Matrix_eWiseMult ...
                    (C, Mask, accum, op, AT, BT, dtt);
                C1 = GB_mex_Matrix_eWiseMult ...
                    (C, Mask, accum, op, AT, BT, dtt);
                GB_spec_compare (C0, C1, 0, tol) ;

                if (track_coverage)
                    c = sum (GraphBLAS_grbcov > 0) ;
                    d = c - clast ;
                    if (d == 0)
                        fprintf ('.', d) ;
                    else
                        fprintf ('[%d]', d) ;
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
        end

    end
end

fprintf ('\ntest18: all tests passed\n') ;

