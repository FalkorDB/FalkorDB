function test14
%TEST14 test GrB_reduce

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest14: reduce to column and scalar\n') ;

[~, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

rng ('default') ;

m = 8 ;
n = 4 ;
dt = struct ('inp0', 'tran') ;

for k1 = 1:length(types)
    atype = types {k1} ;
    fprintf ('.') ;
    A = GB_spec_random (m, n, 0.3, 100, atype) ;
    B = GB_spec_random (n, m, 0.3, 100, atype) ;
    w = GB_spec_random (m, 1, 0.3, 100, atype) ;
    cin = GB_mex_cast (0, atype) ;

    clear S_input
    S_input.matrix = cin ;
    S_input.pattern = true ;
    S_input.class = atype ;

    clear E_input
    E_input.matrix = sparse (0) ;
    E_input.pattern = false ;
    E_input.class = atype ;

    mask = GB_random_mask (m, 1, 0.5, true, false) ;

    if (isequal (atype, 'logical'))
        ops = {'or', 'and', 'xor', 'eq', 'any'} ;
    else
        ops = {'min', 'max', 'plus', 'times', 'any'} ;
    end

    if (isequal (atype, 'double'))
        hrange = [0 1] ;
        crange = [0 1] ;
    else
        hrange = 0 ;
        crange = 1 ;
    end

    is_float = test_contains (atype, 'single') || test_contains (atype, 'double') ;

    for A_is_hyper = 0:1
    for A_is_csc   = 0:1

    A.is_csc = A_is_csc ; A.is_hyper = A_is_hyper ;
    B.is_csc = A_is_csc ; B.is_hyper = A_is_hyper ;

    for k2 = 1:length(add_ops)
        op = add_ops {k2} ;

        if (isequal (op, 'any'))
            tol = [ ] ;
        elseif (test_contains (atype, 'single'))
            tol = 1e-5 ;
        elseif (test_contains (atype, 'double'))
            tol = 1e-12 ;
        else
            tol = 0 ;
        end

        try
            GB_spec_operator (op, atype) ;
            identity = GB_spec_identity (op, atype) ;
        catch
            continue
        end

        % no mask
        w1 = GB_spec_reduce_to_vector (w, [], [], op, A, []) ;
        w2 = GB_mex_reduce_to_vector  (w, [], [], op, A, []) ;
        GB_spec_compare (w1, w2, identity, tol) ;

        % no mask, with accum
        w1 = GB_spec_reduce_to_vector (w, [], 'plus', op, A, []) ;
        w2 = GB_mex_reduce_to_vector  (w, [], 'plus', op, A, []) ;
        GB_spec_compare (w1, w2, identity, tol) ;

        % with mask
        w1 = GB_spec_reduce_to_vector (w, mask, [], op, A, []) ;
        w2 = GB_mex_reduce_to_vector  (w, mask, [], op, A, []) ;
        GB_spec_compare (w1, w2, identity, tol) ;

        % with mask and accum
        w1 = GB_spec_reduce_to_vector (w, mask, 'plus', op, A, []) ;
        w2 = GB_mex_reduce_to_vector  (w, mask, 'plus', op, A, []) ;
        GB_spec_compare (w1, w2, identity, tol) ;

        % no mask, transpose
        w1 = GB_spec_reduce_to_vector (w, [], [], op, B, dt) ;
        w2 = GB_mex_reduce_to_vector  (w, [], [], op, B, dt) ;
        GB_spec_compare (w1, w2, identity, tol) ;

        % no mask, with accum, transpose
        w1 = GB_spec_reduce_to_vector (w, [], 'plus', op, B, dt) ;
        w2 = GB_mex_reduce_to_vector  (w, [], 'plus', op, B, dt) ;
        GB_spec_compare (w1, w2, identity, tol) ;

        % with mask, transpose
        w1 = GB_spec_reduce_to_vector (w, mask, [], op, B, dt) ;
        w2 = GB_mex_reduce_to_vector  (w, mask, [], op, B, dt) ;
        GB_spec_compare (w1, w2, identity, tol) ;

        % with mask and accum, transpose
        w1 = GB_spec_reduce_to_vector (w, mask, 'plus', op, B, dt) ;
        w2 = GB_mex_reduce_to_vector  (w, mask, 'plus', op, B, dt) ;
        GB_spec_compare (w1, w2, identity, tol) ;

        % GB_spec_reduce_to_scalar always operates column-wise, but GrB_reduce
        % operates in whatever order it is given: by column if CSC or by row if
        % CSR.  The result can vary slightly because of different round off
        % errors.  A_flip causes GB_spec_reduce_to_scalar to operate in the
        % same order as GrB_reduce.

        A_flip = A ;
        if (~A.is_csc && is_float)
            A_flip.matrix = A_flip.matrix.' ;
            A_flip.pattern = A_flip.pattern' ;
            A_flip.is_csc = true ;
        end

        % Parallel reduction leads to different roundoff.  So even with A_flip,
        % c1 and c2 can only be compared to within round-off error.

        % to scalar
        c2 = GB_mex_reduce_to_scalar  (cin, [ ], op, A) ;
        if (isequal (op, 'any'))
            X = GB_mex_cast (full (A.matrix (A.pattern)), A.class) ;
            assert (any (X == c2)) ;
        else
            c1 = GB_spec_reduce_to_scalar (cin, [ ], op, A_flip) ;
            if (is_float)
                assert (abs (c1-c2) < tol *  (abs(c1) + 1))
            else
                assert (isequal (c1, c2)) ;
            end
        end

        % to GrB_Scalar
        S = GB_mex_reduce_to_GrB_Scalar (S_input, [ ], op, A) ;
        c2 = S.matrix ;
        if (isequal (op, 'any'))
            X = GB_mex_cast (full (A.matrix (A.pattern)), A.class) ;
            assert (any (X == c2)) ;
        else
            c1 = GB_spec_reduce_to_scalar (cin, [ ], op, A_flip) ;
            if (is_float)
                assert (abs (c1-c2) < tol *  (abs(c1) + 1))
            else
                assert (isequal (c1, c2)) ;
            end
        end

        % to GrB_Scalar
        S = GB_mex_reduce_to_GrB_Scalar (E_input, [ ], op, A) ;
        c2 = S.matrix ;
        if (isequal (op, 'any'))
            X = GB_mex_cast (full (A.matrix (A.pattern)), A.class) ;
            assert (any (X == c2)) ;
        else
            c1 = GB_spec_reduce_to_scalar (cin, [ ], op, A_flip) ;
            if (is_float)
                assert (abs (c1-c2) < tol *  (abs(c1) + 1))
            else
                assert (isequal (c1, c2)) ;
            end
        end

        % vector to GrB_Scalar
        S = GB_mex_reduce_to_GrB_Scalar (S_input, [ ], op, w) ;
        c2 = S.matrix ;
        if (isequal (op, 'any'))
            X = GB_mex_cast (full (w.matrix (w.pattern)), w.class) ;
            assert (any (X == c2)) ;
        else
            c1 = GB_spec_reduce_to_scalar (cin, [ ], op, w) ;
            if (is_float)
                assert (abs (c1-c2) < tol *  (abs(c1) + 1))
            else
                assert (isequal (c1, c2)) ;
            end
        end

        % to scalar, with accum
        c2 = GB_mex_reduce_to_scalar (cin, 'plus', op, A) ;
        if (~isequal (op, 'any'))
            c1 = GB_spec_reduce_to_scalar (cin, 'plus', op, A_flip) ;
            if (is_float)
                assert (abs (c1-c2) < tol *  (abs(c1) + 1))
            else
                assert (isequal (c1, c2)) ;
            end
        end

        % to GrB_Scalar, with accum
        S = GB_mex_reduce_to_GrB_Scalar (S_input, 'plus', op, A) ;
        c2 = S.matrix ;
        if (~isequal (op, 'any'))
            c1 = GB_spec_reduce_to_scalar (cin, 'plus', op, A_flip) ;
            if (is_float)
                assert (abs (c1-c2) < tol *  (abs(c1) + 1))
            else
                assert (isequal (c1, c2)) ;
            end
        end

        % vector to GrB_Scalar, with accum
        S = GB_mex_reduce_to_GrB_Scalar (S_input, 'plus', op, w) ;
        c2 = S.matrix ;
        if (~isequal (op, 'any'))
            c1 = GB_spec_reduce_to_scalar (cin, 'plus', op, w) ;
            if (is_float)
                assert (abs (c1-c2) < tol *  (abs(c1) + 1))
            else
                assert (isequal (c1, c2)) ;
            end
        end

    end
    end
    end
end

clear A
A.matrix = sparse (4,5) ;
A.pattern = false (4,5) ;
A.class = 'double' ;

clear S_input
S_input.matrix = 1 ;
S_input.pattern = true ;
S_input.class = 'double' ;

% empty matrix to GrB_Scalar
S = GB_mex_reduce_to_GrB_Scalar (S_input, [ ], 'plus', A) ;
assert (nnz (S.matrix) == 0) ;

fprintf ('\ntest14: all tests passed\n') ;

