function test243
%TEST243 test GxB_Vector_Iterator

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;
ntypes = length (types) ;

n = 30 ;
GB_builtin_complex_set (true) ;

% scalar_in is dense and all zero
scalar_in.matrix = sparse (0) ;
scalar_in.pattern = true ;

desc.inp0 = 'tran' ;

for k = 1:(ntypes + 1)

    if (k == ntypes + 1)
        % user-defined double complex
        GB_builtin_complex_set (false) ;
        type = 'double complex'  ;
        fprintf ('\nComplex (UDT):\n') ;
    else
        % built-in types
        type = types {k}  ;
        fprintf ('\n%s:\n', type) ;
    end
    
    if (test_contains (type, 'single'))
        tol = 1e-5 ;
    elseif (test_contains (type, 'double'))
        tol = 1e-11 ;
    else
        tol = 0 ;
    end

    X = [ ] ;
    Y = [ ] ;

    if (isequal (type, 'bool'))
        accum.opname = 'lor' ;
        semiring.add = 'lor' ;
        semiring.multiply = 'land' ;
    else
        accum.opname = 'plus' ;
        semiring.add = 'plus' ;
        semiring.multiply = 'times' ;
    end
    accum.optype = type ;
    semiring.class = type ;

    ntrials = 51 ;

    for X_sparsity = [2 4 8]

        if (X_sparsity == 8)
            fprintf ('X full: ') ;
        elseif (X_sparsity == 4)
            fprintf ('X bitmap: ') ;
        elseif (X_sparsity == 2)
            fprintf ('X sparse: ') ;
        end

        for Y_sparsity = [2 4 8]

            if (Y_sparsity == 8)
                fprintf ('Y full') ;
            elseif (Y_sparsity == 4)
                fprintf ('Y bitmap') ;
            elseif (Y_sparsity == 2)
                fprintf ('Y sparse') ;
            end

            for trial = 1:ntrials

                fprintf ('.') ;
                if (X_sparsity == 8 || trial >= 50)
                    X = GB_spec_random (n, 1, inf, 100, type) ;
                else
                    d = trial / 200 ;
                    X = GB_spec_random (n, 1, d, 100, type) ;
                end
                X.sparsity = X_sparsity ;
                X.is_csc = true ;

                if (Y_sparsity == 8 || trial == 51)
                    Y = GB_spec_random (n, 1, inf, 100, type) ;
                else
                    d = trial / 200 ;
                    Y = GB_spec_random (n, 1, d, 100, type) ;
                end
                Y.sparsity = Y_sparsity ;
                Y.is_csc = true ;

                % scalar is empty
                scalar_in.class = type ;
                scalar_in.iso = false ;

                % y0 = X'*Y
                y0 = GB_spec_mxm (scalar_in, [ ], accum, semiring, X, Y, desc) ;

                % with vector iterator, with macros
                y1 = GB_mex_dot_iterator (X, Y, 0) ;
                GB_spec_compare (y0, y1, 0, tol) ;

                % with vector iterator, brute force, with macros
                y1 = GB_mex_dot_iterator (X, Y, 1) ;
                GB_spec_compare (y0, y1, 0, tol) ;

                % with vector iterator, with functions
                y1 = GB_mex_dot_iterator (X, Y, 2) ;
                GB_spec_compare (y0, y1, 0, tol) ;

                % with vector iterator, brute force, with functions
                y1 = GB_mex_dot_iterator (X, Y, 3) ;
                GB_spec_compare (y0, y1, 0, tol) ;

            end
            fprintf ('\n') ;
        end
    end
end
GB_builtin_complex_set (true) ;
fprintf ('\ntest243: all tests passed\n') ;

