function test134(short)
%TEST134 test GxB_select

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% A shorter version of test25

fprintf ('\ntest134: GxB_select tests\n') ;

if (nargin < 1)
    short = false ;
end

[~, ~, ~, types, ~, select_ops] = GB_spec_opsall ;
types = types.all ;

if (short)
    types = { 'double' } ;
end

rng ('default') ;

m = 10 ;
n = 6 ;
dt = struct ('inp0', 'tran') ;

for k1 = 1:length(types)
    atype = types {k1} ;
    fprintf ('%-14s ', atype) ;

    for A_sparsity = [0 1 2]
    for A_is_csc   = 0:1
    for C_is_hyper = 0:1
    for C_is_csc   = 0:1
    for M_is_hyper = 0:1
    for M_is_csc   = 0:1

    if (A_sparsity == 0)
        A_is_hyper = 0 ;
        A_sparsity_control = 2 ;    % sparse
    elseif (A_sparsity == 1)
        A_is_hyper = 1 ;
        A_sparsity_control = 1 ;    % hypersparse
    else
        A_is_hyper = 0 ;
        A_sparsity_control = 4 ;    % bitmap
    end

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
    Mask = GB_random_mask (m, n, 0.5, M_is_csc, M_is_hyper) ;
    Mask.hyper_switch = hm ;

    A.sparsity = A_sparsity_control ;
    B.sparsity = A_sparsity_control ;

    fprintf ('.') ;

    for k2 = 1:length(select_ops)
        op = select_ops {k2} ;

        if (test_contains (atype, 'complex'))
            switch (op)
                case { 'gt_zero', 'ge_zero', 'lt_zero', 'le_zero', ...
                       'gt_thunk', 'ge_thunk', 'lt_thunk', 'le_thunk' }
                    continue ;
                    % error ('op %s not defined for complex types', op) ;
                otherwise
                    % op is OK
            end
        end

        switch op
            case {'tril'    }
                klist = [-4 0 4] ;
            case {'triu'    }
                klist = [-4 0 4] ;
            case {'diag'    }
                klist = [-4 0 4] ;
            case {'offdiag' }
                klist = [-4 0 4] ;
            case {'nonzero' }
                klist = 0 ;
            case {'eq_zero' }
                klist = 0 ;
            case {'gt_zero' }
                klist = 0 ;
            case {'ge_zero' }
                klist = 0 ;
            case {'lt_zero' }
                klist = 0 ;
            case {'le_zero' }
                klist = 0 ;
            case {'ne_thunk'}
                klist = [0 1] ;
            case {'eq_thunk'}
                klist = [0 1] ;
            case {'gt_thunk'}
                klist = [0 1] ;
            case {'ge_thunk'}
                klist = [0 1] ;
            case {'lt_thunk'}
                klist = [0 1] ;
            case {'le_thunk'}
                klist = [0 1] ;
        end



        for k = klist

            % no mask
            C1 = GB_spec_select (Cin, [], [], op, A, k, []) ;
            C2 = GB_mex_select  (Cin, [], [], op, A, k, [], 'test') ;
            GB_spec_compare (C1, C2) ;

            % no mask, transpose
            C1 = GB_spec_select (Cin, [], [], op, B, k, dt) ;
            C2 = GB_mex_select  (Cin, [], [], op, B, k, dt, 'test') ;
            GB_spec_compare (C1, C2) ;

            if (strcmp (op, 'tril'))

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
    end
    end
    fprintf ('\n') ;

end

fprintf ('\ntest134: all tests passed\n') ;


