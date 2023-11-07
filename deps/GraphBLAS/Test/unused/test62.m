function test62
%TEST62 test GrB_apply

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n ------------ testing GrB_apply\n') ;

rng ('default')

[binops, unary_ops, ~, types, ~, ~] = GB_spec_opsall ;
ops = unary_ops.all ;
accum_ops = binops.all ;
types = types.all ;

dt = struct ('inp0', 'tran') ;

% type of the matrix C
for k1 = 1:length (types)
    ctype = types {k1}  ;
    fprintf ('\n%s', ctype) ;

    % type of the matrix A
    for k2 = 1:length (types)
        atype = types {k2} ;

        % create a matrix
        for m = [1 10 25]
            for n = [1 10 25]
                fprintf ('.') ;
                clear A
                A.matrix = sprandn (m, n, 0.1) ;
                A.class = atype ;

                Mask = (sprandn (m, n, 0.1) ~= 0) ;
                MaskT = Mask' ;

                clear B
                B.matrix = sprandn (m*n, 1, 0.1) ;
                B.class = atype ;

                mask = (sprandn (m*n, 1, 0.1) ~= 0) ;

                clear Cin
                Cin.matrix = sprandn (m, n, 0.1) ;
                Cin.class = ctype ;

                clear CinT
                CinT.matrix = Cin.matrix' ;
                CinT.class = ctype ;

                clear Cin2
                Cin2.matrix = sprandn (m*n, 1, 0.1) ;
                Cin2.class = ctype ;

                % unary operator
                for k3 = 1:length(ops)
                    unary_op = ops {k3}  ;
                    ntypes = 1;length (types) ;
                    % fprintf ('unary: %s\n', unary_op) ;
                    % unary operator type
                    for k4 = ntypes
                        clear unary
                        if (~isempty (unary_op))
                            unary_type = types {k4}  ;
                            unary.opname = unary_op ;
                            unary.optype = unary_type ;
                        else
                            unary = '' ;
                            unary_type = '' ;
                        end

                        try
                            GB_spec_operator (unary_op) ;
                        catch
                            continue
                        end

                        % accum operator
                        for k5 = 0:length(accum_ops)
                            if (k5 == 0)
                                accum_op = ''  ;
                                ntypes = 1 ;
                            else
                                accum_op = accum_ops {k5}  ;
                                ntypes = 1;length (types) ;
                            end
                            % accum operator type
                            for k6 = ntypes
                                clear accum
                                if (~isempty (accum_op))
                                    accum_type = types {k6}  ;
                                    accum.opname = accum_op ;
                                    accum.optype = accum_type ;
                                else
                                    accum = '' ;
                                    accum_type = '' ;
                                end

                                if (GB_spec_is_positional (accum))
                                    continue ;
                                end

                                try
                                    GB_spec_operator (accum) ;
                                catch
                                    continue
                                end

                                % apply to A, no mask
                                C1 = GB_mex_apply  (Cin, [ ], accum, unary, A, [ ]) ;
                                C2 = GB_spec_apply (Cin, [ ], accum, unary, A, [ ]) ;
                                GB_spec_compare (C1, C2) ;

                                % apply to A', no mask
                                C2 = GB_spec_apply (CinT, [ ], accum, unary, A, dt) ;
                                C1 = GB_mex_apply  (CinT, [ ], accum, unary, A, dt) ;
                                GB_spec_compare (C1, C2) ;

                                % apply to vector B, no mask
                                C1 = GB_mex_apply  (Cin2, [ ], accum, unary, B, [ ]) ;
                                C2 = GB_spec_apply (Cin2, [ ], accum, unary, B, [ ]) ;
                                GB_spec_compare (C1, C2) ;

                                % apply to A, with mask
                                C1 = GB_mex_apply  (Cin, Mask, accum, unary, A, [ ]) ;
                                C2 = GB_spec_apply (Cin, Mask, accum, unary, A, [ ]) ;
                                GB_spec_compare (C1, C2) ;

                                % apply to A', with mask
                                C1 = GB_mex_apply  (CinT, MaskT, accum, unary, A, dt) ;
                                C2 = GB_spec_apply (CinT, MaskT, accum, unary, A, dt) ;
                                GB_spec_compare (C1, C2) ;

                                % apply to vector B, with mask
                                C1 = GB_mex_apply  (Cin2, mask, accum, unary, B, [ ]) ;
                                C2 = GB_spec_apply (Cin2, mask, accum, unary, B, [ ]) ;
                                GB_spec_compare (C1, C2) ;

                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf ('\ntest62: all tests passed\n') ;

