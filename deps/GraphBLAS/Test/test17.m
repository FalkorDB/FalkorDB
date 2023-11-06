function test17
%TEST17 test GrB_*_extractElement

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n ------------ testing GrB_extractElement\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

rng ('default') ;

% type of the output X
for k1 = 1:length (types)
    xtype = types {k1}  ;
    fprintf ('\n%-14s ', xtype) ;

    % type of the matrix A
    for k2 = 1:length (types)
        atype = types {k2}  ;
        fprintf ('.') ;

        % create a matrix
        for m = [1 10] % [1 10 25 50]
            for n = [1 10] % [1 10 25 50]
                clear A
                A.matrix = 100 * sprandn (m, n, 0.1) ;
                A.matrix (1,1) = pi ;
                A.class = atype ;

                clear B
                B.matrix = 100 * sprandn (m*n, 1, 0.1) ;
                B.matrix (1,1) = sparse (0) ;
                B.class = atype ;

                for A_is_hyper = 0:1
                for A_is_csc   = 0:1
                A.is_hyper = A_is_hyper ;
                A.is_csc   = A_is_csc   ;
                for i = 0:m-1
                    iu = uint64 (i) ;
                    for j = 0:n-1
                        ju = uint64 (j) ;
                        use_scalar = (rand (1) > 0.9) ;
                        x1 = GB_mex_Matrix_extractElement  (A, iu, ju, ...
                                xtype, use_scalar) ;
                        [x2 noval] = ...
                            GB_spec_Matrix_extractElement (A, i, j, xtype) ;
                        assert (isequal (x1,x2))
                        x3 = GB_mex_Matrix_isStoredElement (A, iu, ju) ;
                        assert (isequal (~noval,x3))
                    end
                end
                end
                end

                for i = 0:(m*n)-1
                    iu = uint64 (i) ;
                    use_scalar = (rand (1) > 0.9) ;
                    x1 = GB_mex_Vector_extractElement  (B, iu, xtype, ...
                        use_scalar) ;
                    [x2 noval] = ...
                        GB_spec_Vector_extractElement (B, i, xtype) ;
                    assert (isequal (x1,x2))
                    x3 = GB_mex_Vector_isStoredElement (B, iu) ;
                    assert (isequal (~noval,x3))
                end
            end
        end
    end
end

fprintf ('\ntest17: all tests passed\n') ;

