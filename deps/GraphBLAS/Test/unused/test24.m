function test24(fulltest)
%TEST24 test GrB_reduce
% test24(fulltest); fulltest=1 if longer test, 0 for quick test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
test_types = types.all ;
accum_ops = binops.all ;

rng ('default') ;

if (nargin < 1)
    fulltest = 0 ;
end

if (fulltest)
    fprintf ('\ntest24: ----- exhaustive test of GrB_reduce_to_scalar and vector\n') ;
    cset = 1:length (test_types) ;
    aset = 1:length (test_types) ;
else
    fprintf ('\ntest24: ----- quick test of GrB_reduce_to_scalar and vector\n');
    cset = 10 ;
    aset = 11 ;
end

dt = struct ('inp0', 'tran') ;

% class of the vector x
for k1 = cset
    cclass = test_types {k1}  ;
    cin = GB_mex_cast (2, cclass) ;
    % fprintf ('\n===================================  c class: %s\n',cclass) ;

    % class of the matrix A
    for k2 = aset
        aclass = test_types {k2}  ;
        % fprintf ('\n==================================A class: %s\n',aclass) ;
        fprintf ('[%s %s]', cclass, aclass) ;

        % create a matrix
        for m = [1 10 25]
            for n = [1 10 25]
                fprintf ('.') ;
                clear A
                A.matrix = sprandn (m, n, 0.1) ;
                A.class = aclass ;

                clear B
                B.matrix = sprandn (m*n, 1, 0.1) ;
                B.class = aclass ;

                clear xin
                xin.matrix = sprandn (m, 1, 0.1) ;
                xin.class = cclass ;

                clear yin
                yin.matrix = sprandn (n, 1, 0.1) ;
                yin.class = cclass ;

                % reduce operator
                for k3 = 1:length(add_ops)
                    if (k3 == 0)
                        reduce_op = ''  ;
                        ntypes = 1 ;
                    else
                        reduce_op = add_ops {k3}  ;
                        ntypes = 1;length (test_types) ;
                    end
                    % fprintf ('reduce: %s\n', reduce_op) ;
                    % reduce operator class

                    for k4 = ntypes
                        clear reduce
                        if (~isempty (reduce_op))
                            reduce_class = test_types {k4}  ;
                            reduce.opname = reduce_op ;
                            reduce.optype = reduce_class ;
                        else
                            reduce = '' ;
                            reduce_class = '' ;
                        end

                        try
                            [opname optype ztype xtype ytype] = ...
                                GB_spec_operator (reduce) ;
                        catch
                            continue
                        end

                        if (~isequal (reduce_class, 'logical') && ...
                            ( isequal (reduce_op, 'or') || ...
                              isequal (reduce_op, 'and') || ...
                              isequal (reduce_op, 'xor') || ...
                              isequal (reduce_op, 'eq')))
                            continue ;
                        end

                        identity = GB_spec_identity (reduce) ;

                        % accum operator
                        for k5 = 0:length(accum_ops)
                            if (k5 == 0)
                                accum_op = ''  ;
                                ntypes = 1 ;
                            else
                                accum_op = accum_ops {k5}  ;
                                ntypes = 1;length (test_types) ;
                            end
                            % accum operator class
                            for k6 = ntypes
                                clear accum
                                if (~isempty (accum_op))
                                    accum_class = test_types {k6}  ;
                                    accum.opname = accum_op ;
                                    accum.optype = accum_class ;
                                else
                                    accum = '' ;
                                    accum_class = '' ;
                                end

                                if (GB_spec_is_positional (accum))
                                    continue ;
                                end

                                try
                                    [opname optype ztype xtype ytype] = GB_spec_operator (accum) ;
                                catch
                                    continue
                                end

                                % reduce matrix to scalar
                                c  = GB_mex_reduce_to_scalar  (cin, accum, reduce, A) ;
                                c3 = GB_spec_reduce_to_scalar (cin, accum, reduce, A) ;
                                assert (isequal (c, c3))

                                % reduce vector to scalar
                                c  = GB_mex_reduce_to_scalar  (cin, accum, reduce, B) ;
                                c3 = GB_spec_reduce_to_scalar (cin, accum, reduce, B) ;
                                assert (isequal (c, c3))

                                % row-wise reduce matrix to vector

                                % no mask
                                x  = GB_mex_reduce_to_vector  (xin, [ ], accum, reduce, A, [ ]) ;
                                x3 = GB_spec_reduce_to_vector (xin, [ ], accum, reduce, A, [ ]) ;
                                GB_spec_compare (x, x3, identity) ;

                                % with mask
                                mask = sprandn (m,1,0.3) ~= 0 ;
                                x  = GB_mex_reduce_to_vector  (xin, mask, accum, reduce, A, [ ]) ;
                                x3 = GB_spec_reduce_to_vector (xin, mask, accum, reduce, A, [ ]) ;
                                GB_spec_compare (x, x3, identity) ;

                                % col-wise reduce matrix to vector

                                % no mask
                                y  = GB_mex_reduce_to_vector  (yin, [ ], accum, reduce, A, dt) ;
                                y3 = GB_spec_reduce_to_vector (yin, [ ], accum, reduce, A, dt) ;
                                GB_spec_compare (y, y3, identity) ;

                                % with mask
                                mask = sprandn (n,1,0.3) ~= 0 ;
                                y  = GB_mex_reduce_to_vector  (yin, mask, accum, reduce, A, dt) ;
                                y3 = GB_spec_reduce_to_vector (yin, mask, accum, reduce, A, dt) ;
                                GB_spec_compare (y, y3, identity) ;

                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf ('\ntest24: all tests passed\n') ;

