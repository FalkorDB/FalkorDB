function test29
%TEST29 GrB_reduce with zombies

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

fprintf ('\ntest29: ----------------- GrB_reduce with zombies\n') ;

for m = [1 5 10]
    for n = [1 5 10]

        fprintf ('.') ;
        rng ('default') ;

        for k3 = 1:length (types)
            atype = types {k3}  ;

            for builtin = 0:1
            GB_builtin_complex_set (builtin) ;

            clear C
            C.matrix = 100 * sparse (rand (m,n)) ;
            C.class = atype ;
            C.pattern = logical (spones (C.matrix)) ;

            for A_is_hyper = 0:1
            for A_is_csc   = 0:1

            A = GB_spec_random (m,n,0.1,100,atype, A_is_csc, A_is_hyper) ;

            for kk4 = 1:length(add_ops)

                op = add_ops {kk4} ;

                if (~builtin)
                    % no user-defined Complex_any_monoid
                    if (test_contains (atype, 'complex'))
                        if (isequal (op, 'any'))
                            continue ;
                        end
                    end
                end

                try
                    GB_spec_operator (op, atype) ;
                    GB_builtin_complex_set (true) ;
                    cin = GB_spec_identity (op, atype) ;
                    GB_builtin_complex_set (builtin) ;
                catch
                    continue
                end

                if (isempty (cin))
                    GB_builtin_complex_set (true) ;
                    cin = GB_mex_cast (0, atype) ;
                    GB_builtin_complex_set (builtin) ;
                end

                [C3,c1,c3] = GB_mex_subassign (C, [ ], [ ], A, ...
                    [ ], [ ], [ ], op) ;
                c2 = GB_mex_reduce_to_scalar (cin, '', op, C3) ;

                if (isequal (op, 'any'))
                    [i,j,x] = find (C3.matrix) ;
                    if (length (x) == 0)
                        assert (c1 == 0) ;
                        assert (c2 == 0) ;
                    else
                        assert (any (c1 == x)) ;
                        assert (any (c2 == x)) ;
                    end
                elseif (isfloat (c1))
                    assert (isequal (c1,c2) || ...
                        (abs (c1-c2) <= 8 * eps (c2)))  ;
                else
                    assert (isequal (c1,c2))
                end

                if (~builtin)
                    % optype is double, which can't be used to
                    % reduce the user-defined type 'Complex'
                    if (test_contains (atype, 'complex'))
                        continue ;
                    end
                end

                op_plus.opname = 'plus' ;
                op_plus.optype = 'double' ;
                c4 = GB_mex_reduce_to_scalar (0, '', op_plus, C3) ;
                if (isfloat (c3))
                    assert (isequal (c3,c4) || ...
                        (abs (c3-c4) <= 8 * eps (c4)))  ;
                else
                    assert (isequal (c3,c4))
                end

            end
            end
            end
            end
        end
    end
end

GB_builtin_complex_set (true) ;
fprintf ('\ntest29: all tests passed\n') ;

