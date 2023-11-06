function test18(fulltest)
%TEST18 test GrB_eWiseAdd, GxB_eWiseUnion, and GrB_eWiseMult

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
bin_ops = binops.all ;
types = types.all ;

if (nargin < 1)
    fulltest = 0 ;
end

if (fulltest == 2)
    fprintf ('test18 --------lengthy tests of GrB_eWiseAdd, eWiseUnion, eWiseMult\n') ;
    k1test = 1:length(types) ;
    k4test = randi([0,length(bin_ops)])
    k6list = [false true] ;
    k7list = [false true] ;
    k8list = 0:1 ;
elseif (fulltest == 1)
    fprintf ('test18 ------------tests of GrB_eWiseAdd, eWiseUnion, and eWiseMult\n') ;
    % k1test = [1 2 4 10 11] ;
    k1test = [ 1 2 11 ] ;
    k4test = randi([0,length(bin_ops)])
    k6list = [false true] ;
    k7list = [false true] ;
    k8list = 0:1 ;
else
    fprintf ('test18 ------quick tests of GrB_eWiseAdd, eWiseUnion, and eWiseMult\n') ;
    k1test = [ 1 2 11 12 13] ;
    k4test = 0 ;
    k6list = [false] ;
    k7list = [false] ;
    k8list = 1 ;
end

%   mlist = [1 5 10] ; 
%   nlist = [1 5 10] ; 
    mlist = [10] ; 
    nlist = [10] ; 

rng ('default') ;

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

n_semirings = 0 ;
for k1 = k1test % 1:length (types)
    type = types {k1}  ;

    fprintf ('\n\n========================================= %s:\n', type) ;

    k2test = 1:length(bin_ops) ;

    for k2 = k2test % 1:length(bin_ops)
        binop = bin_ops {k2}  ;

        fprintf ('\n%s', binop) ;

        op.opname = binop ;
        op.optype = type ;

        try
            GB_spec_operator (op) ;
        catch
            continue
        end

        fprintf (' binary op: [ %s %s ] ', binop, type) ;
        if (test_contains (type, 'single'))
            tol = 1e-5 ;
        elseif (test_contains (type, 'double'))
            tol = 1e-12 ;
        else
            tol = 0 ;
        end

        % try some matrices
        for m = mlist
            for n = nlist

                % avoid creating nans for testing pow
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

                % these tests do not convert real A and B into complex C for C
                % = A.^B.  So ensure the test matrices are all positive.
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

                for k4 = k4test

                    clear accum
                    if (k4 == 0)
                        accum = ''  ;
                        ntypes = 1 ;
                    else
                        accum.opname = bin_ops {k4}  ;
                        ntypes = length (types) ;
                        fprintf ('accum: %s ', accum.opname) ;
                    end

                    if (GB_spec_is_positional (accum))
                        continue
                    end

                    for k5 = randi ([1 ntypes]) % ntypes

                        if (k4 > 0)
                            accum.optype = types {k5}  ;

                            try
                                GB_spec_operator (accum) ;
                            catch
                                continue
                            end

                            fprintf ('%s\n', accum.optype) ;
                        else
                            fprintf ('\n') ;
                        end

                        for Mask_complement = k6list

                            if (Mask_complement)
                                dnn.mask = 'complement' ;
                                dtn.mask = 'complement' ;
                                dnt.mask = 'complement' ;
                                dtt.mask = 'complement' ;
                            else
                                dnn.mask = 'default' ;
                                dtn.mask = 'default' ;
                                dnt.mask = 'default' ;
                                dtt.mask = 'default' ;
                            end

                            for C_replace = k7list

                                if (C_replace)
                                    dnn.outp = 'replace' ;
                                    dtn.outp = 'replace' ;
                                    dnt.outp = 'replace' ;
                                    dtt.outp = 'replace' ;
                                else
                                    dnn.outp = 'default' ;
                                    dtn.outp = 'default' ;
                                    dnt.outp = 'default' ;
                                    dtt.outp = 'default' ;
                                end

                                for A_is_hyper = 0:1
                                for A_is_csc   = 0:1
                                for B_is_hyper = 0:1
                                for B_is_csc   = 0:1
                                fprintf ('.') ;
                                for C_is_hyper = 0 % 0:1
                                for C_is_csc   = 0 % 0:1

                                for native = k8list

                                clear A
                                A.matrix = Amat ;
                                A.is_hyper = A_is_hyper ;
                                A.is_csc   = A_is_csc   ;
                                if (native)
                                    A.class = op.optype ;
                                end

                                clear AT
                                AT.matrix = ATmat ;
                                AT.is_hyper = A_is_hyper ;
                                AT.is_csc   = A_is_csc   ;
                                if (native)
                                    AT.class = op.optype ;
                                end

                                clear B
                                B.matrix = Bmat ;
                                B.is_hyper = B_is_hyper ;
                                B.is_csc   = B_is_csc   ;
                                if (native)
                                    B.class = op.optype ;
                                end

                                clear BT
                                BT.matrix = BTmat ;
                                BT.is_hyper = B_is_hyper ;
                                BT.is_csc   = B_is_csc   ;
                                if (native)
                                    BT.class = op.optype ;
                                end

                                clear C
                                C.matrix = Cmat ;
                                C.is_hyper = C_is_hyper ;
                                C.is_csc   = C_is_csc   ;

                                clear u
                                u.matrix = uvec ;
                                u.is_csc = true ;
                                if (native)
                                    u.class = op.optype ;
                                end

                                clear v
                                v.matrix = vvec ;
                                v.is_csc = true ;
                                if (native)
                                    v.class = op.optype ;
                                end

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

                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf ('\ntest18: all tests passed\n') ;

