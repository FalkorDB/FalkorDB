function test129
%TEST129 test GxB_select (tril and nonzero, hypersparse)

% This is a shorter version of test25

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest129: GxB_select tests (tril and nonzero)\n') ;

[~, ~, ~, ~, ~, select_ops] = GB_spec_opsall ;

rng ('default') ;

if (~ispc)
    % This test is not done on Windows, since the Complex type is 
    % always the same as the built-in GxB_FC64, and not user-defined.
    fprintf ('\n---------- Trigger an intentional error (domain mismatch):\n\n') ;
    GB_builtin_complex_set (0) ;    % use the user-defined Complex type
    try
        % this must fail; the scalar thunk cannot be user-defined for tril
        C = sparse (1i) ;
        C = GB_mex_select (C, [ ], [ ], 'tril', C, C, [ ]) ;
        % ack! The call to GB_mex_select was supposed to have failed.
        ok = false ;
    catch me
        % GB_mex_select correctly returned an error
        fprintf ('Intentional error: %s\n', me.message) ;
        ok = true ;
    end
    assert (ok) ;
    fprintf ('---------- Domain mismatch error above is expected\n\n') ;
    GB_builtin_complex_set (1) ;    % use the built-in GxB_FC64 Complex type
end

m = 10 ;
n = 6 ;
dt = struct ('inp0', 'tran') ;

    atype = 'double' ;

    for A_is_hyper = 0:1
    for A_is_csc   = 0:1
    for C_is_hyper = 0:1

    for C_is_csc   = 0:1
    for M_is_hyper = 0:1
    for M_is_csc   = 0:1

    if (A_is_hyper)
        ha = 1 ;
    else
        ha = 0 ;
    end

    if (C_is_hyper)
        hc = 1 ;
    else
        hc = 0 ;
    end

    if (M_is_hyper)
        hm = 1 ;
    else
        hm = 0 ;
    end

    A = GB_spec_random (m, n, 0.3, 100, atype, A_is_csc, A_is_hyper, ha) ;
    A.matrix (:,1) = rand (m,1) ;
    A.pattern (:,1) = true (m,1) ;
    Cin = GB_spec_random (m, n, 0.3, 100, atype, C_is_csc, C_is_hyper, hc) ;
    B = GB_spec_random (n, m, 0.3, 100, atype, A_is_csc, A_is_hyper, ha) ;
    cin = GB_mex_cast (0, atype) ;
    % Mask = (sprand (m, n, 0.5) ~= 0) ;
    Mask = GB_random_mask (m, n, 0.5, M_is_csc, M_is_hyper) ;
    Mask.hyper_switch = hm ;

    fprintf ('.') ;

    for k2 = [ 1 5 ]
        op = select_ops {k2} ;

        k = sparse (0) ;

            % no mask
            C1 = GB_spec_select (Cin, [], [], op, A, k, []) ;
            C2 = GB_mex_select  (Cin, [], [], op, A, k, [], 'test') ;
            GB_spec_compare (C1, C2) ;

            % no mask, with accum
            C1 = GB_spec_select (Cin, [], 'plus', op, A, k, []) ;
            C2 = GB_mex_select  (Cin, [], 'plus', op, A, k, [], 'test') ;
            GB_spec_compare (C1, C2) ;

            % with mask
            C1 = GB_spec_select (Cin, Mask, [], op, A, k, []) ;
            C2 = GB_mex_select  (Cin, Mask, [], op, A, k, [], 'test') ;
            GB_spec_compare (C1, C2) ;

            % with mask and accum
            C1 = GB_spec_select (Cin, Mask, 'plus', op, A, k, []) ;
            C2 = GB_mex_select  (Cin, Mask, 'plus', op, A, k, [], 'test') ;
            GB_spec_compare (C1, C2) ;

            % no mask, transpose
            C1 = GB_spec_select (Cin, [], [], op, B, k, dt) ;
            C2 = GB_mex_select  (Cin, [], [], op, B, k, dt, 'test') ;
            GB_spec_compare (C1, C2) ;

            % no mask, with accum, transpose
            C1 = GB_spec_select (Cin, [], 'plus', op, B, k, dt) ;
            C2 = GB_mex_select  (Cin, [], 'plus', op, B, k, dt, 'test') ;
            GB_spec_compare (C1, C2) ;

            % with mask, transpose
            C1 = GB_spec_select (Cin, Mask, [], op, B, k, dt) ;
            C2 = GB_mex_select  (Cin, Mask, [], op, B, k, dt, 'test') ;
            GB_spec_compare (C1, C2) ;

            % with mask and accum, transpose
            C1 = GB_spec_select (Cin, Mask, 'plus', op, B, k, dt) ;
            C2 = GB_mex_select  (Cin, Mask, 'plus', op, B, k, dt, 'test') ;
            GB_spec_compare (C1, C2) ;

    end

    end
    end
    end
    end
    end
    end

GB_builtin_complex_set (true) ;
fprintf ('\ntest129: all tests passed\n') ;

