function test127
%TEST127 test GrB_eWiseAdd and GrB_eWiseMult (all types and operators)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
binops = binops.all ;
types = types.all ;

fprintf ('test127 -----------tests of GrB_eWiseAdd and eWiseMult (all ops)\n') ;

m = 5 ;
n = 5 ;

rng ('default') ;

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;
dnn_notM = struct ('mask', 'complement') ;

Amat2 = sparse (2 * sprand (m,n, 0.8)) ;
Bmat2 = sparse (2 * sprand (m,n, 0.8)) ;
Cmat2 = sparse (2 * sprand (m,n, 0.8)) ;
w2 = sparse (2 * sprand (m,1, 0.8)) ;
uvec2 = sparse (2 * sprand (m,1, 0.8)) ;
vvec2 = sparse (2 * sprand (m,1, 0.8)) ;

Amat = sparse (100 * sprandn (m,n, 0.8)) ;
Bmat = sparse (100 * sprandn (m,n, 0.8)) ;
Cmat = sparse (100 * sprandn (m,n, 0.8)) ;
w = sparse (100 * sprandn (m,1, 0.8)) ;
uvec = sparse (100 * sprandn (m,1, 0.8)) ;
vvec = sparse (100 * sprandn (m,1, 0.8)) ;

Maskmat = sprandn (m,n,0.9) ~= 0 ;
maskvec = sprandn (m,1,0.9) ~= 0 ;

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

ATmat2 = Amat2.' ;
BTmat2 = Bmat2.' ;

for k1 = 1:length (types)
    type = types {k1}  ;
    fprintf ('\n\n%-8s : ', type) ;

    for k2 = 1:length(binops)
        binop = binops {k2}  ;

        op.opname = binop ;
        op.optype = type ;

        if (test_contains (type, 'single'))
            tol = 1e-5 ;
        elseif (test_contains (type, 'double'))
            tol = 1e-12 ;
        else
            tol = 0 ;
        end
        
        try
            GB_spec_operator (op) ;
        catch
            continue ;
        end

        fprintf (' %s', binop) ;

        for A_sparsity_control = 0:1
        for A_is_csc   = 0 % 0:1
        for B_sparsity_control = 0:1
        for B_is_csc   = 0 % 0:1
        for C_sparsity_control = 0:1
        for C_is_csc   = 0 % 0:1

        if (A_sparsity_control == 0)
            A_is_hyper = 0 ; % not hyper
            A_sparsity = 1 ; % sparse
        else
            A_is_hyper = 0 ; % not hyper
            A_sparsity = 4 ; % bitmap
        end

        if (B_sparsity_control == 0)
            B_is_hyper = 0 ; % not hyper
            B_sparsity = 1 ; % sparse
        else
            B_is_hyper = 0 ; % not hyper
            B_sparsity = 4 ; % bitmap
        end

        if (C_sparsity_control == 0)
            C_is_hyper = 0 ; % not hyper
            C_sparsity = 1 ; % sparse
        else
            C_is_hyper = 0 ; % not hyper
            C_sparsity = 4 ; % bitmap
        end

        for native = 1 % 0:1

        clear A AT B BT C u v

        if (isequal (binop, 'pow'))
            A.matrix = Amat2 ;
            AT.matrix = ATmat2 ;
            B.matrix = Bmat2 ;
            BT.matrix = BTmat2 ;
            C.matrix = Cmat2 ;
            u.matrix = uvec2 ;
            v.matrix = vvec2 ;
        else
            A.matrix = Amat ;
            AT.matrix = ATmat ;
            B.matrix = Bmat ;
            BT.matrix = BTmat ;
            C.matrix = Cmat ;
            u.matrix = uvec ;
            v.matrix = vvec ;
        end

        A.is_hyper = A_is_hyper ;
        A.is_csc   = A_is_csc   ;
        A.sparsity = A_sparsity ;
        if (native)
            A.class = op.optype ;
        end

        AT.is_hyper = A_is_hyper ;
        AT.sparsity = A_sparsity ;
        AT.is_csc   = A_is_csc   ;
        if (native)
            AT.class = op.optype ;
        end

        B.is_hyper = B_is_hyper ;
        B.sparsity = B_sparsity ;
        B.is_csc   = B_is_csc   ;
        if (native)
            B.class = op.optype ;
        end

        BT.is_hyper = B_is_hyper ;
        BT.sparsity = B_sparsity ;
        BT.is_csc   = B_is_csc   ;
        if (native)
            BT.class = op.optype ;
        end

        C.is_hyper = C_is_hyper ;
        C.is_csc   = C_is_csc   ;
        C.sparsity = C_sparsity ;

        u.is_csc = true ;
        if (native)
            u.class = op.optype ;
        end

        v.is_csc = true ;
        if (native)
            v.class = op.optype ;
        end

        %---------------------------------------
        % A+B
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (C, [ ], [ ], op, A, B, dnn) ;
        C1 = GB_mex_Matrix_eWiseAdd  (C, [ ], [ ], op, A, B, dnn) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        w0 = GB_spec_Vector_eWiseAdd (w, [ ], [ ], op, u, v, dnn) ;
        w1 = GB_mex_Vector_eWiseAdd  (w, [ ], [ ], op, u, v, dnn) ;
        GB_spec_compare (w0, w1, 0, tol) ;

        %---------------------------------------
        % A'+B
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (C, [ ], [ ], op, AT, B, dtn) ;
        C1 = GB_mex_Matrix_eWiseAdd  (C, [ ], [ ], op, AT, B, dtn) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A+B'
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (C, [ ], [ ], op, A, BT, dnt) ;
        C1 = GB_mex_Matrix_eWiseAdd  (C, [ ], [ ], op, A, BT, dnt) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A'+B'
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (C, [ ], [ ], op, AT, BT, dtt) ;
        C1 = GB_mex_Matrix_eWiseAdd  (C, [ ], [ ], op, AT, BT, dtt) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A.*B
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op, A, B, dnn) ;
        C1 = GB_mex_Matrix_eWiseMult  (C, [ ], [ ], op, A, B, dnn) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        w0 = GB_spec_Vector_eWiseMult (w, [ ], [ ], op, u, v, dnn) ;
        w1 = GB_mex_Vector_eWiseMult  (w, [ ], [ ], op, u, v, dnn) ;
        GB_spec_compare (w0, w1, 0, tol) ;

        %---------------------------------------
        % A'.*B
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op, AT, B, dtn) ;
        C1 = GB_mex_Matrix_eWiseMult  (C, [ ], [ ], op, AT, B, dtn) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % B.*A'
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op, B, AT, dnt) ;
        C1 = GB_mex_Matrix_eWiseMult  (C, [ ], [ ], op, B, AT, dnt) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A.*B'
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op, A, BT, dnt) ;
        C1 = GB_mex_Matrix_eWiseMult  (C, [ ], [ ], op, A, BT, dnt) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A'.*B'
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseMult (C, [ ], [ ], op, AT, BT, dtt) ;
        C1 = GB_mex_Matrix_eWiseMult  (C, [ ], [ ], op, AT, BT, dtt) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %-----------------------------------------------
        % with mask
        %-----------------------------------------------

        for M_is_very_sparse = 0 % 0:1
        % for M_is_hyper = 0 % 0:1
        for M_sparsity_control = 0:1
        for M_is_csc   = 0 % 0:1

        clear Mask mask
        if (M_is_very_sparse)
            Mask.matrix = Maskmat2 ;
            mask.matrix = maskvec2 ;
        else
            Mask.matrix = Maskmat ;
            mask.matrix = maskvec ;
        end

        if (M_sparsity_control == 0)
            M_is_hyper = 0 ; % not hyper
            M_sparsity = 1 ; % sparse
        else
            M_is_hyper = 0 ; % not hyper
            M_sparsity = 4 ; % bitmap
        end

        Mask.is_hyper = M_is_hyper ;
        Mask.sparsity = M_sparsity ;
        Mask.is_csc   = M_is_csc   ;
        mask.is_csc = true ;

        %---------------------------------------
        % A+B, with mask
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (C, Mask, [ ], op, A, B, dnn) ;
        C1 = GB_mex_Matrix_eWiseAdd  (C, Mask, [ ], op, A, B, dnn) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        w0 = GB_spec_Vector_eWiseAdd (w, mask, [ ], op, u, v, dnn) ;
        w1 = GB_mex_Vector_eWiseAdd  (w, mask, [ ], op, u, v, dnn) ;
        GB_spec_compare (w0, w1, 0, tol) ;

        %---------------------------------------
        % A+B, with mask complemented
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (C, Mask, [ ], op, A, B, dnn_notM) ;
        C1 = GB_mex_Matrix_eWiseAdd  (C, Mask, [ ], op, A, B, dnn_notM) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        w0 = GB_spec_Vector_eWiseAdd (w, mask, [ ], op, u, v, dnn_notM) ;
        w1 = GB_mex_Vector_eWiseAdd  (w, mask, [ ], op, u, v, dnn_notM) ;
        GB_spec_compare (w0, w1, 0, tol) ;

        %---------------------------------------
        % A'+B, with mask
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (C, Mask, [ ], op, AT, B, dtn) ;
        C1 = GB_mex_Matrix_eWiseAdd  (C, Mask, [ ], op, AT, B, dtn) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A+B', with mask
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (C, Mask, [ ], op, A, BT, dnt) ;
        C1 = GB_mex_Matrix_eWiseAdd  (C, Mask, [ ], op, A, BT, dnt) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A'+B', with mask
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseAdd (C, Mask, [ ], op, AT, BT, dtt) ;
        C1 = GB_mex_Matrix_eWiseAdd  (C, Mask, [ ], op, AT, BT, dtt) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A.*B, with mask
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseMult (C, Mask, [ ], op, A, B, dnn) ;
        C1 = GB_mex_Matrix_eWiseMult  (C, Mask, [ ], op, A, B, dnn) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        w0 = GB_spec_Vector_eWiseMult (w, mask, [ ], op, u, v, dnn) ;
        w1 = GB_mex_Vector_eWiseMult  (w, mask, [ ], op, u, v, dnn) ;
        GB_spec_compare (w0, w1, 0, tol) ;

        %---------------------------------------
        % A'.*B, with mask
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseMult (C, Mask, [ ], op, AT, B, dtn) ;
        C1 = GB_mex_Matrix_eWiseMult  (C, Mask, [ ], op, AT, B, dtn) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A.*B', with mask
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseMult (C, Mask, [ ], op, A, BT, dnt) ;
        C1 = GB_mex_Matrix_eWiseMult  (C, Mask, [ ], op, A, BT, dnt) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A'.*B', with mask
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseMult (C, Mask, [ ], op, AT, BT, dtt) ;
        C1 = GB_mex_Matrix_eWiseMult  (C, Mask, [ ], op, AT, BT, dtt) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        %---------------------------------------
        % A.*B, with mask complemented
        %---------------------------------------

        C0 = GB_spec_Matrix_eWiseMult (C, Mask, [ ], op, A, B, dnn_notM) ;
        C1 = GB_mex_Matrix_eWiseMult  (C, Mask, [ ], op, A, B, dnn_notM) ;
        GB_spec_compare (C0, C1, 0, tol) ;

        w0 = GB_spec_Vector_eWiseMult (w, mask, [ ], op, u, v, dnn_notM) ;
        w1 = GB_mex_Vector_eWiseMult  (w, mask, [ ], op, u, v, dnn_notM) ;
        GB_spec_compare (w0, w1, 0, tol) ;


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

fprintf ('\ntest127: all tests passed\n') ;

