function test10
%TEST10 test GrB_apply

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest10: GrB_apply tests\n') ;

[~, unary_ops, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;
unary_ops = unary_ops.all ;

rng ('default') ;

m = 8 ;
n = 4 ;
dt = struct ('inp0', 'tran') ;
dr = struct ('outp', 'replace') ;

for k1 = 1:length(types)
    atype = types {k1} ;
    fprintf ('\n%s: ', atype) ;

    Mask = GB_random_mask (m, n, 0.5, true, false) ;
    Cin = GB_spec_random (m, n, 0.3, 100, atype) ;
    Cmask = spones (GB_mex_cast (full (Cin.matrix), Cin.class)) ;

    % for most operators 
    A = GB_spec_random (m, n, 0.3, 100, atype) ;
    B = GB_spec_random (n, m, 0.3, 100, atype) ;

    A_matrix = A.matrix ;
    B_matrix = B.matrix ;

    % for pow, sqrt, log, log10, log2, gammaln (domain is [0,inf])
    A_pos_matrix = abs (A.matrix) ;
    B_pos_matrix = abs (B.matrix) ;

    % for asin, acos, atanh (domain is [-1,1])
    A_1_matrix = A_matrix ;
    B_1_matrix = B_matrix ;
    A_1_matrix (abs (A_matrix) > 1) = 1 ;
    B_1_matrix (abs (B_matrix) > 1) = 1 ;

    % for acosh, asech (domain is [1, inf])
    A_1inf_matrix = A_matrix ;
    B_1inf_matrix = B_matrix ;
    A_1inf_matrix (A_matrix < 1 & A_matrix ~= 0) = 1 ;
    B_1inf_matrix (B_matrix < 1 & B_matrix ~= 0) = 1 ;

    % for log1p (domain is [-1, inf])
    A_n1inf_matrix = A_matrix ;
    B_n1inf_matrix = B_matrix ;
    A_n1inf_matrix (A_matrix < -1) = 1 ;
    B_n1inf_matrix (B_matrix < -1) = 1 ;

    % for tanh: domain is [-inf,inf], but rounding to
    % integers fails when x is outside this range
    A_5_matrix = A_matrix ;
    B_5_matrix = B_matrix ;
    A_5_matrix (abs (A_matrix) > 5) = 5 ;
    B_5_matrix (abs (B_matrix) > 5) = 5 ;

    % for gamma: domain is [-inf,inf], but not defined for negative
    % integers, and rounding to integers fails when x is outside this range
    A_pos5_matrix = A_matrix ;
    B_pos5_matrix = B_matrix ;
    A_pos5_matrix (A_matrix <= 0.1 & A_matrix ~= 0) = 0.1 ;
    B_pos5_matrix (B_matrix <= 0.1 & B_matrix ~= 0) = 0.1 ;
    A_pos5_matrix (A_matrix > 5) = 5 ;
    B_pos5_matrix (B_matrix > 5) = 5 ;

%{
    % for minv: do not try 1/0 for the complex case
    A_nonzero_matrix = A_matrix ;
    B_nonzero_matrix = B_matrix ;
    A_pattern = full (logical (spones (A_matrix))) ;
    B_pattern = full (logical (spones (B_matrix))) ;
    A_zero = full (abs (A_nonzero_matrix) < 1) ;
    B_zero = full (abs (B_nonzero_matrix) < 1) ;
    A_nonzero_matrix (A_pattern & A_zero) = 1 ;
    B_nonzero_matrix (B_pattern & B_zero) = 1 ;
%}

    % do longer tests for a few types
    longer_tests = isequal (atype, 'double') || isequal (atype, 'int64') ;
    if (longer_tests)
        hrange = [0 1] ;
        crange = [0 1] ;
    else
        hrange = 0 ;
        crange = 1 ;
    end

    for k2 = 1:length(unary_ops)
        op.opname = unary_ops {k2} ;
        if (longer_tests)
            fprintf ('\n') ;
        end
        fprintf (' %s', op.opname) ;

        for k3 = 1:length(types)
            op.optype = types {k3} ;

            if (ispc && test_contains (op.opname, 'asin') && test_contains (op.optype, 'complex'))
                % casin and casinf are broken on Windows
                fprintf ('#') ;
                continue ;
            end

            if (ismac && test_contains (op.opname, 'log') ...
                && test_contains (op.optype, 'complex') ...
                && (test_contains (atype, 'int') || isequal (atype, 'logical')))
                % complex log2 on the mac has a slight O(eps) roundoff that
                % results in a different result when typecasted to integer
                fprintf ('#') ;
                continue ;
            end

            try
                [opname optype ztype xtype ytype] = GB_spec_operator (op) ;
            catch
                continue 
            end
            fprintf ('.') ;

            A.matrix = A_matrix ;
            B.matrix = B_matrix ;

            switch (opname)
                % domain is ok, but limit it to avoid integer typecast
                % failures from O(eps) errors, or overflow to inf
                case { 'tanh', 'exp', 'sin', 'cos', 'tan', ...
                    'sinh', 'cosh', 'asin', 'acos', 'acosh', 'asinh', ...
                    'atanh', 'exp2', 'expm1', 'carg', 'atan' }
                    A.matrix = A_5_matrix ;
                    B.matrix = B_5_matrix ;
                case { 'tgamma' }
                    A.matrix = A_pos5_matrix ;
                    B.matrix = B_pos5_matrix ;
                otherwise
                    % no change
            end

            if (~test_contains (optype, 'complex'))

                % for real operators, avoiding complex results
                switch (opname)
                    case { 'pow', 'sqrt', 'log', 'log10', 'log2', ...
                        'gammaln', 'lgamma' }
                        A.matrix = A_pos_matrix ;
                        B.matrix = B_pos_matrix ;
                    case { 'asin', 'acos', 'atanh' }
                        A.matrix = A_1_matrix ;
                        B.matrix = B_1_matrix ;
                    case { 'acosh', 'asech' }
                        A.matrix = A_1inf_matrix ;
                        B.matrix = B_1inf_matrix ;
                    case 'log1p'
                        A.matrix = A_n1inf_matrix ;
                        B.matrix = B_n1inf_matrix ;
                    case { 'tanh', 'exp' }
                        % domain is ok, but limit it to avoid integer typecast
                        % failures from O(eps) errors
                        A.matrix = A_5_matrix ;
                        B.matrix = B_5_matrix ;
                    case { 'minv' }
%                       A.matrix = A_nonzero_matrix ;
%                       B.matrix = B_nonzero_matrix ;
                    otherwise
                        % no change
                end

            else

                % operator is complex
                switch (opname)
                    % domain is ok, but limit it to avoid integer typecast
                    % failures from O(eps) errors
                    case { 'log2' }
                        A.matrix = A_5_matrix ;
                        B.matrix = B_5_matrix ;
                    otherwise
                        % no change
                end

            end

            % op

            tol = 0 ;
            if (test_contains (optype, 'single') || test_contains (atype, 'single'))
                tol = 1e-5 ;
            elseif (test_contains (optype, 'double') || test_contains (atype, 'double'))
                tol = 1e-12 ;
            end

            for A_sparsity = [hrange 2]

            if (A_sparsity == 0)
                A_is_hyper = 0 ;
                A_is_bitmap = 0 ;
                A_sparsity_control = 2 ;    % sparse
            elseif (A_sparsity == 1)
                A_is_hyper = 1 ;
                A_is_bitmap = 0 ;
                A_sparsity_control = 1 ;    % hypersparse
            else
                A_is_hyper = 0 ;
                A_is_bitmap = 1 ;
                A_sparsity_control = 4 ;    % bitmap
            end

            for A_is_csc   = crange

            if (longer_tests)
                fprintf ('.') ;
            end

            for C_is_hyper = hrange
            for C_is_csc   = crange
            for M_is_hyper = hrange
            for M_is_csc   = crange
            A.is_csc    = A_is_csc ; A.is_hyper    = A_is_hyper ;
            Cin.is_csc  = C_is_csc ; Cin.is_hyper  = C_is_hyper ;
            B.is_csc    = A_is_csc ; B.is_hyper    = A_is_hyper ;
            Mask.is_csc = M_is_csc ; Mask.is_hyper = M_is_hyper ;

            A.sparsity = A_sparsity_control ;
            B.sparsity = A_sparsity_control ;

            % no mask
            C1 = GB_spec_apply (Cin, [], [], op, A, []) ;
            C2 = GB_mex_apply  (Cin, [], [], op, A, []) ;
            test10_compare (op, C1, C2, tol) ;

            % with mask
            C1 = GB_spec_apply (Cin, Mask, [], op, A, []) ;
            C2 = GB_mex_apply  (Cin, Mask, [], op, A, []) ;
            test10_compare (op, C1, C2, tol) ;

            % with C == mask, and outp = replace
            C1 = GB_spec_apply (Cin, Cmask, [], op, A, dr) ;
            C2 = GB_mex_apply_maskalias (Cin, [], op, A, dr) ;
            test10_compare (op, C1, C2, tol) ;

            % no mask, transpose
            C1 = GB_spec_apply (Cin, [], [], op, B, dt) ;
            C2 = GB_mex_apply  (Cin, [], [], op, B, dt) ;
            test10_compare (op, C1, C2, tol) ;

            % with mask, transpose
            C1 = GB_spec_apply (Cin, Mask, [], op, B, dt) ;
            C2 = GB_mex_apply  (Cin, Mask, [], op, B, dt) ;
            test10_compare (op, C1, C2, tol) ;

            switch (opname)
                % the results from these operators must be check before summing
                % their results with the accum operator, so skip the rest of
                % the tests.
                case { 'acos', 'asin', 'atan' 'acosh', 'asinh', 'atanh' }
                    continue ;
            end

            % no mask, with accum
            C1 = GB_spec_apply (Cin, [], 'plus', op, A, []) ;
            C2 = GB_mex_apply  (Cin, [], 'plus', op, A, []) ;
            test10_compare (op, C1, C2, tol) ;

            % with mask and accum
            C1 = GB_spec_apply (Cin, Mask, 'plus', op, A, []) ;
            C2 = GB_mex_apply  (Cin, Mask, 'plus', op, A, []) ;
            test10_compare (op, C1, C2, tol) ;

            % with C == mask and accum, and outp = replace
            C1 = GB_spec_apply (Cin, Cmask, 'plus', op, A, dr) ;
            C2 = GB_mex_apply_maskalias (Cin, 'plus', op, A, dr) ;
            test10_compare (op, C1, C2, tol) ;

            % no mask, with accum, transpose
            C1 = GB_spec_apply (Cin, [], 'plus', op, B, dt) ;
            C2 = GB_mex_apply  (Cin, [], 'plus', op, B, dt) ;
            test10_compare (op, C1, C2, tol) ;

            % with mask and accum, transpose
            C1 = GB_spec_apply (Cin, Mask, 'plus', op, B, dt) ;
            C2 = GB_mex_apply  (Cin, Mask, 'plus', op, B, dt) ;
            test10_compare (op, C1, C2, tol) ;

        end
    end

    end
    end
    end
    end
    end
    end
    fprintf ('\n') ;

end

fprintf ('\ntest10: all tests passed\n') ;

