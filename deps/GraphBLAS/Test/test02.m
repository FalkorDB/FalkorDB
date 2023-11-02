function test02
%TEST02 test GrB_*_dup

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
GB_builtin_complex_set (true) ;
types = types.all ;

rng ('default') ;
format long g

for k1 = 1:length (types)
    atype = types {k1} ;

    for is_hyper = 0:1
        for is_csc = 0:1

            % create a random A
            A = GB_spec_random (4, 4, 0.8, 128, atype, is_csc, is_hyper) ;
            A.matrix (1,1) = -1 ;
            A.pattern (1,1) = true ;
            A_matrix = full (A.matrix) ;
            A_pattern = full (A.pattern) ;
            assert (GB_spok (1*A.matrix) == 1) ;
            assert (GB_spok (A.pattern) == 1) ;

            for k2 = 1:length (types)
                ctype = types {k2} ;
                % typecast to type of C

                C = GB_mex_dup (A, ctype) ;
                C_matrix = full (C.matrix) ;
                C_pattern = full (GB_spones_mex (C.matrix)) ;
                assert (GB_spok (1*C.matrix) == 1) ;

                if (k1 == k2)
                    % also try another method
                    assert (isequal (A_pattern, C_pattern)) ;
                    assert (isequal (A.class, C.class)) ;

                    C2 = GB_mex_dup (A, ctype, 1) ;
                    C2_matrix = full (C2.matrix) ;
                    C2_pattern = full (GB_spones_mex (C2.matrix)) ;
                    assert (isequal (C, C2))  ;
                    assert (GB_spok (1*C2.matrix) == 1) ;
                end

            end
        end
    end
end

% try with both built-in and user-defined 'double complex' types:
for k = [false true]
    GB_builtin_complex_set (k) ;

    % duplicate a complex matrix (user-defined can't be typecasted)
    A = GB_mex_random (4, 4, 10, 1) ;
    assert (GB_spok (1*A) == 1) ;

    C = GB_mex_dup (A) ;
    % C_matrix = full (C.matrix) ;
    assert (isequal (A, C.matrix))  ;
    assert (GB_spok (1*C.matrix) == 1) ;

    C = GB_mex_dup (A, 'double complex', 1) ;
    % C_matrix = full (C.matrix) ;
    assert (isequal (A, C.matrix))  ;
    assert (GB_spok (1*C.matrix) == 1) ;
end

format

GB_builtin_complex_set (true) ;
fprintf ('test02: all typecast and copy tests passed\n') ;

