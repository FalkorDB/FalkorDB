function test03
%TEST03 test GB_*_check functions

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

rng ('default') ;

for k = 1:length (types)
    aclass = types {k} ;
    for is_hyper = 0:1
        for is_csc = 0:1
            A = GB_spec_random (10,30,0.2,100,aclass, is_csc, is_hyper) ;
            C = GB_mex_dump (A,2) ;
            GB_spec_compare (C, A) ;
        end
    end
    for hyper_switch = -0.1:0.1:0.4
        A = GB_spec_random (10,30,0.02,100,aclass, is_csc, [ ], hyper_switch) ;
        C = GB_mex_dump (A,2) ;
        GB_spec_compare (C, A) ;
    end
end

for is_hyper = 0:1
    for is_csc = 0:1
        A = GB_spec_random (100,2,0.5,100,'int8', is_csc, is_hyper) ;
        C = GB_mex_dump (A,2) ;
        GB_spec_compare (C, A) ;
    end
end

for k = [false true]
    fprintf ('builtin_complex: %d\n', k) ;
    GB_builtin_complex_set (k) ;

    % complex case:
    A = GB_mex_random (10, 30, 15, 1, 1, 0, 0, 0) ;
    C = GB_mex_dump (A,2) ;
    GB_spec_compare (C, A) ;
    A = GB_mex_random (10, 30, 15, 1, 1, 0, 0, 1) ;
    C = GB_mex_dump (A,2) ;
    GB_spec_compare (C, A) ;
    A = GB_mex_random (10, 30, 15, 1, 1, 1, 0, 1) ;
    C = GB_mex_dump (A,2) ;
    GB_spec_compare (C, A) ;
    A = GB_mex_random (10, 30, 15, 1, 1, 1, 0, 0) ;
    C = GB_mex_dump (A,2) ;
    GB_spec_compare (C, A) ;
    A = GB_mex_random (10, 30, 15, 1, 1, 1, 1, 1) ;
    C = GB_mex_dump (A,2) ;
    GB_spec_compare (C, A) ;
    A = GB_mex_random (3, 3, 5, 0, 1, 1, 1, 3) 
    C = GB_mex_dump (A)
    GB_spec_compare (C, A) ;
end

GB_builtin_complex_set (true) ;
fprintf ('\ntest03: all object check tests passed\n') ;

